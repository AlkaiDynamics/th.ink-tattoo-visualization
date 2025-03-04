"""
Server-side tattoo generation API for the Th.ink AR application.

This module provides the FastAPI routes and handlers for generating and
manipulating tattoo designs using AI models, with features for various
tattoo styles, customization options, and caching.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Union, Literal
import os
import logging
import time
import asyncio
import uuid
import json
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO

from ..database import models, schemas, dependencies
from ..auth.security import get_current_active_user
from .model_handler import generate_tattoo_image, get_model_handler
from ..errors.error_handler import ModelError, ValidationError
from ..config.model_config import get_config

# Configure logger
logger = logging.getLogger("think.server.ai.tattoo_generator")

# Initialize router
router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)

# Get configuration
config = get_config()

# Setup paths
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("static/tattoos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Request models
class TattooGeneratorRequest(BaseModel):
    """Request model for tattoo generation."""
    description: str = Field(
        ..., 
        description="Text description of the desired tattoo",
        example="A phoenix rising from ashes with flames in red and orange"
    )
    style: Optional[str] = Field(
        "traditional", 
        description="Tattoo style to generate",
        example="traditional"
    )
    size: Optional[Literal["small", "medium", "large"]] = Field(
        "medium",
        description="Size of the tattoo design"
    )
    color: Optional[bool] = Field(
        True,
        description="Whether to generate a color tattoo"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible generation"
    )
    advanced_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Advanced generation options"
    )

    @validator('description')
    def validate_description(cls, v):
        """Validate the description length."""
        if len(v) < 3:
            raise ValueError("Description must be at least 3 characters")
        if len(v) > 500:
            raise ValueError("Description must be at most 500 characters")
        return v
    
    @validator('style')
    def validate_style(cls, v):
        """Validate the tattoo style."""
        valid_styles = [
            "traditional", "realistic", "blackwork", 
            "watercolor", "tribal", "japanese", 
            "new_school", "minimalist", "geometric", "dotwork"
        ]
        if v.lower() not in valid_styles:
            raise ValueError(f"Style must be one of: {', '.join(valid_styles)}")
        return v.lower()


class TattooAdjustmentRequest(BaseModel):
    """Request model for tattoo design adjustments."""
    design_id: Optional[int] = Field(
        None,
        description="ID of the existing design to adjust (either design_id or temp_id required)"
    )
    temp_id: Optional[str] = Field(
        None,
        description="Temporary ID of the design to adjust"
    )
    adjustments: Dict[str, Any] = Field(
        ...,
        description="Adjustment parameters"
    )
    
    @validator('adjustments')
    def validate_adjustments(cls, v):
        """Validate adjustment parameters."""
        valid_keys = ['rotation', 'scale', 'color_adjustments', 'filter', 'transparency']
        for key in v.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid adjustment key: {key}")
        
        if 'rotation' in v and not isinstance(v['rotation'], (int, float)):
            raise ValueError("Rotation must be a number")
        
        if 'scale' in v and not isinstance(v['scale'], (int, float)):
            raise ValueError("Scale must be a number")
        
        if 'transparency' in v and not 0 <= v['transparency'] <= 1:
            raise ValueError("Transparency must be between 0 and 1")
        
        return v


class BlendDesignsRequest(BaseModel):
    """Request model for blending two designs."""
    design_id1: Optional[int] = Field(
        None,
        description="ID of the first design to blend"
    )
    design_id2: Optional[int] = Field(
        None,
        description="ID of the second design to blend"
    )
    temp_id1: Optional[str] = Field(
        None,
        description="Temporary ID of the first design to blend"
    )
    temp_id2: Optional[str] = Field(
        None,
        description="Temporary ID of the second design to blend"
    )
    blend_ratio: float = Field(
        0.5,
        description="Blend ratio between designs (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    blend_mode: str = Field(
        "normal",
        description="Blending mode"
    )
    
    @validator('blend_mode')
    def validate_blend_mode(cls, v):
        """Validate blend mode."""
        valid_modes = ["normal", "multiply", "screen", "overlay"]
        if v not in valid_modes:
            raise ValueError(f"Blend mode must be one of: {', '.join(valid_modes)}")
        return v


# Response models
class TattooGeneratorResponse(BaseModel):
    """Response model for tattoo generation."""
    image_url: str = Field(
        ...,
        description="URL to the generated tattoo image"
    )
    design_id: Optional[int] = Field(
        None,
        description="ID of the saved design, if available"
    )
    temp_id: str = Field(
        ...,
        description="Temporary ID for the design"
    )
    generation_time: float = Field(
        ...,
        description="Time taken to generate the design in seconds"
    )
    size: Dict[str, int] = Field(
        ...,
        description="Image dimensions (width, height)"
    )
    metadata: Dict[str, Any] = Field(
        {},
        description="Additional generation metadata"
    )


@router.post("/tattoo-generator", response_model=TattooGeneratorResponse)
async def generate_tattoo(
    request: TattooGeneratorRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Generate a tattoo image based on the provided description.
    
    This endpoint uses AI to create tattoo designs based on text prompts.
    """
    try:
        # Log the request
        logger.info(f"Generating tattoo for user {current_user.id}: {request.description[:30]}...")
        
        # Check user limits based on subscription
        subscription_tier = current_user.subscription_tier
        usage_tracker = UsageTracker(db)
        
        # Check if user has reached their generation limit
        if not usage_tracker.can_generate(current_user.id, subscription_tier):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Daily generation limit reached for your subscription tier"
            )
        
        # Start timing
        start_time = time.time()
        
        # Convert size to dimensions
        size_dimensions = {
            "small": (512, 512),
            "medium": (768, 768),
            "large": (1024, 1024)
        }
        dimensions = size_dimensions.get(request.size, (768, 768))
        
        # Prepare options
        options = request.advanced_options or {}
        if request.seed is not None:
            options['seed'] = request.seed
        
        # Generate the tattoo image
        try:
            image_bytes, metadata = await generate_tattoo_image(
                request.description,
                style=request.style,
                size=request.size,
                color=request.color,
                **options
            )
        except ModelError as e:
            logger.error(f"Tattoo generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate tattoo: {str(e)}"
            )
        
        # Create a temporary ID for the generated image
        temp_id = str(uuid.uuid4())
        
        # Save the generated image
        if image_bytes:
            # Create temporary file path
            temp_path = TEMP_DIR / f"{temp_id}.png"
            
            # Save the image
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            
            # Create a public URL
            image_url = f"/static/temp/{temp_id}.png"
            
            # Setup cleanup task to run after response is sent
            # Remove the temporary file after 24 hours
            background_tasks.add_task(
                cleanup_temp_file, 
                temp_path, 
                delay_hours=24
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate tattoo image"
            )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Update user's generation count
        usage_tracker.record_generation(current_user.id)
        
        # Get image dimensions
        img = Image.open(BytesIO(image_bytes))
        size = {"width": img.width, "height": img.height}
        
        return {
            "image_url": image_url,
            "design_id": None,  # Not saved to database yet
            "temp_id": temp_id,
            "generation_time": generation_time,
            "size": size,
            "metadata": metadata
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating tattoo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.post("/save-design", response_model=schemas.TattooDesignRead)
async def save_design(
    temp_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(None),
    price: float = Form(0.0),
    is_public: bool = Form(True),
    tags: List[str] = Form([]),
    style: str = Form("traditional"),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Save a generated design to the database.
    
    This endpoint allows users to save temporary designs permanently.
    """
    try:
        # Check if the temporary file exists
        temp_path = TEMP_DIR / f"{temp_id}.png"
        if not temp_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Temporary design not found"
            )
        
        # Create a permanent path for the design
        design_id = str(uuid.uuid4())
        design_path = OUTPUT_DIR / f"{design_id}.png"
        
        # Copy the file to the permanent location
        import shutil
        shutil.copy2(temp_path, design_path)
        
        # Create design URL
        design_url = f"/static/tattoos/{design_id}.png"
        
        # Get image dimensions
        img = Image.open(design_path)
        width, height = img.size
        
        # Create DB entry for the design
        new_design = models.TattooDesign(
            name=name,
            description=description,
            price=price,
            image_url=design_url,
            artist_id=current_user.id,
            style=style,
            is_public=is_public,
            width=width,
            height=height,
            is_ai_generated=True
        )
        
        db.add(new_design)
        db.commit()
        db.refresh(new_design)
        
        # Add tags if provided
        if tags:
            for tag_name in tags:
                # Find or create tag
                tag = db.query(models.Tag).filter(models.Tag.name == tag_name).first()
                if not tag:
                    tag = models.Tag(name=tag_name)
                    db.add(tag)
                    db.commit()
                    db.refresh(tag)
                
                # Add association
                new_design.tags.append(tag)
            
            db.commit()
            db.refresh(new_design)
        
        return new_design
        
    except Exception as e:
        logger.error(f"Error saving design: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save design: {str(e)}"
        )


@router.post("/adjust-design", response_model=TattooGeneratorResponse)
async def adjust_design(
    request: TattooAdjustmentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Adjust an existing tattoo design.
    
    This endpoint applies transformations like rotation, scaling, and color adjustments.
    """
    try:
        # Check inputs
        if request.design_id is None and request.temp_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either design_id or temp_id must be provided"
            )
        
        # Get the design image
        if request.design_id is not None:
            # Get design from database
            design = db.query(models.TattooDesign).filter(models.TattooDesign.id == request.design_id).first()
            if not design:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Design not found"
                )
            
            # Check if user has access to this design
            if design.artist_id != current_user.id and not design.is_public:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this design"
                )
            
            # Get design image path from URL
            image_path = Path("static") / design.image_url.lstrip("/static/")
            
        else:
            # Get temporary design
            image_path = TEMP_DIR / f"{request.temp_id}.png"
        
        # Check if file exists
        if not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Design image not found"
            )
        
        # Load the image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Get the model handler
        model_handler = get_model_handler()
        
        # Apply adjustments
        try:
            adjusted_image = await model_handler.modify_tattoo_image(
                image_array, 
                request.adjustments
            )
        except Exception as e:
            logger.error(f"Adjustment failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to adjust design: {str(e)}"
            )
        
        # Create a temporary ID for the adjusted image
        temp_id = str(uuid.uuid4())
        
        # Save the adjusted image
        temp_path = TEMP_DIR / f"{temp_id}.png"
        Image.fromarray(adjusted_image).save(temp_path)
        
        # Create a public URL
        image_url = f"/static/temp/{temp_id}.png"
        
        # Setup cleanup task
        background_tasks.add_task(
            cleanup_temp_file, 
            temp_path, 
            delay_hours=24
        )
        
        # Get image dimensions
        size = {"width": adjusted_image.shape[1], "height": adjusted_image.shape[0]}
        
        return {
            "image_url": image_url,
            "design_id": None,  # Not saved to database yet
            "temp_id": temp_id,
            "generation_time": 0.0,  # Not a generation
            "size": size,
            "metadata": {
                "adjustments": request.adjustments,
                "original_design_id": request.design_id,
                "original_temp_id": request.temp_id
            }
        }
        
    except Exception as e:
        logger.error(f"Error adjusting design: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/blend-designs", response_model=TattooGeneratorResponse)
async def blend_designs(
    request: BlendDesignsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Blend two tattoo designs together.
    
    This endpoint combines two designs with various blending modes.
    """
    try:
        # Validate inputs
        if (request.design_id1 is None and request.temp_id1 is None) or \
           (request.design_id2 is None and request.temp_id2 is None):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both designs must be specified (either by design_id or temp_id)"
            )
        
        # Get the first design
        if request.design_id1 is not None:
            design1 = db.query(models.TattooDesign).filter(models.TattooDesign.id == request.design_id1).first()
            if not design1:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="First design not found"
                )
            
            # Check access
            if design1.artist_id != current_user.id and not design1.is_public:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to the first design"
                )
            
            # Get design image path
            image1_path = Path("static") / design1.image_url.lstrip("/static/")
        else:
            # Get temporary design
            image1_path = TEMP_DIR / f"{request.temp_id1}.png"
        
        # Get the second design
        if request.design_id2 is not None:
            design2 = db.query(models.TattooDesign).filter(models.TattooDesign.id == request.design_id2).first()
            if not design2:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Second design not found"
                )
            
            # Check access
            if design2.artist_id != current_user.id and not design2.is_public:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to the second design"
                )
            
            # Get design image path
            image2_path = Path("static") / design2.image_url.lstrip("/static/")
        else:
            # Get temporary design
            image2_path = TEMP_DIR / f"{request.temp_id2}.png"
        
        # Check if files exist
        if not image1_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="First design image not found"
            )
        
        if not image2_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Second design image not found"
            )
        
        # Load the images
        image1 = np.array(Image.open(image1_path))
        image2 = np.array(Image.open(image2_path))
        
        # Get the model handler
        model_handler = get_model_handler()
        
        # Blend the designs
        try:
            blended_image = await model_handler.blend_designs(
                image1,
                image2,
                request.blend_ratio,
                request.blend_mode
            )
        except Exception as e:
            logger.error(f"Blending failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to blend designs: {str(e)}"
            )
        
        # Create a temporary ID for the blended image
        temp_id = str(uuid.uuid4())
        
        # Save the blended image
        temp_path = TEMP_DIR / f"{temp_id}.png"
        Image.fromarray(blended_image).save(temp_path)
        
        # Create a public URL
        image_url = f"/static/temp/{temp_id}.png"
        
        # Setup cleanup task
        background_tasks.add_task(
            cleanup_temp_file, 
            temp_path, 
            delay_hours=24
        )
        
        # Get image dimensions
        size = {"width": blended_image.shape[1], "height": blended_image.shape[0]}
        
        return {
            "image_url": image_url,
            "design_id": None,  # Not saved to database yet
            "temp_id": temp_id,
            "generation_time": 0.0,  # Not a generation
            "size": size,
            "metadata": {
                "blend_ratio": request.blend_ratio,
                "blend_mode": request.blend_mode,
                "design_id1": request.design_id1,
                "design_id2": request.design_id2,
                "temp_id1": request.temp_id1,
                "temp_id2": request.temp_id2
            }
        }
        
    except Exception as e:
        logger.error(f"Error blending designs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/styles", response_model=List[Dict[str, str]])
async def get_tattoo_styles():
    """
    Get all available tattoo styles with descriptions.
    
    This endpoint returns a list of supported tattoo styles.
    """
    # Define available styles with descriptions
    styles = [
        {"id": "traditional", "name": "Traditional", "description": "Bold black outlines with limited color palette"},
        {"id": "realistic", "name": "Realistic", "description": "Photorealistic detail with shading and depth"},
        {"id": "blackwork", "name": "Blackwork", "description": "Solid black ink with intricate patterns"},
        {"id": "watercolor", "name": "Watercolor", "description": "Soft color blending with paint splatter effects"},
        {"id": "tribal", "name": "Tribal", "description": "Bold black tribal patterns with cultural influences"},
        {"id": "japanese", "name": "Japanese", "description": "Traditional Japanese iconography and techniques"},
        {"id": "new_school", "name": "New School", "description": "Cartoonish style with exaggerated proportions and vibrant colors"},
        {"id": "minimalist", "name": "Minimalist", "description": "Clean, simple lines with minimal detail"},
        {"id": "geometric", "name": "Geometric", "description": "Precise geometric shapes and patterns"},
        {"id": "dotwork", "name": "Dotwork", "description": "Intricate patterns of tiny dots creating detailed shading"}
    ]
    
    return styles


@router.get("/usage-limits", response_model=Dict[str, Any])
async def get_usage_limits(
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Get current usage limits and statistics for the authenticated user.
    
    This endpoint returns information about generation limits and usage.
    """
    try:
        # Get usage tracker
        usage_tracker = UsageTracker(db)
        
        # Get subscription tier
        subscription_tier = current_user.subscription_tier
        
        # Get daily limit
        daily_limit = usage_tracker.get_daily_limit(subscription_tier)
        
        # Get current usage
        current_usage = usage_tracker.get_current_usage(current_user.id)
        
        # Get remaining generations
        remaining = max(0, daily_limit - current_usage) if daily_limit >= 0 else -1
        
        # Prepare response
        return {
            "subscription_tier": subscription_tier,
            "daily_limit": daily_limit,
            "current_usage": current_usage,
            "remaining": remaining,
            "unlimited": daily_limit < 0,
            "reset_time": usage_tracker.get_reset_time()
        }
        
    except Exception as e:
        logger.error(f"Error getting usage limits: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage limits"
        )


@router.post("/upload-design", response_model=TattooGeneratorResponse)
async def upload_design(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Upload a custom tattoo design.
    
    This endpoint allows users to upload their own designs for AR preview.
    """
    try:
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PNG and JPEG images are supported"
            )
        
        # Read the file
        contents = await file.read()
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds the 10MB limit"
            )
        
        # Create a temporary ID for the uploaded image
        temp_id = str(uuid.uuid4())
        
        # Save the image
        temp_path = TEMP_DIR / f"{temp_id}.png"
        
        # Process the image to ensure PNG format and alpha channel
        try:
            image = Image.open(BytesIO(contents))
            
            # Convert to RGBA if needed
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Save as PNG
            image.save(temp_path, format="PNG")
            
        except Exception as e:
            logger.error(f"Error processing uploaded image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        # Create a public URL
        image_url = f"/static/temp/{temp_id}.png"
        
        # Setup cleanup task
        background_tasks.add_task(
            cleanup_temp_file, 
            temp_path, 
            delay_hours=24
        )
        
        # Get image dimensions
        size = {"width": image.width, "height": image.height}
        
        return {
            "image_url": image_url,
            "design_id": None,  # Not saved to database yet
            "temp_id": temp_id,
            "generation_time": 0.0,  # Not a generation
            "size": size,
            "metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(contents)
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error uploading design: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded design"
        )


@router.get("/model-status", response_model=Dict[str, Any])
async def get_model_status(
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Get the status of AI models.
    
    This endpoint returns information about the AI models used for tattoo generation.
    """
    try:
        # Get model handler
        model_handler = get_model_handler()
        
        # Get model stats
        model_stats = model_handler.get_model_stats()
        
        return {
            "status": "operational",
            "models": model_stats.get("loaded_models", []),
            "cache_entries": model_stats.get("cache_entries", 0),
            "device": model_stats.get("device", "unknown"),
            "average_generation_time": model_stats.get("overall_avg_time", 0.0),
            "total_calls": model_stats.get("total_calls", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        }


# Helper functions and classes

class UsageTracker:
    """Track and manage user generation limits."""
    
    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db
        
        # Define daily limits by subscription tier
        self.daily_limits = {
            "free": 3,
            "premium": 20,
            "pro": -1  # Unlimited
        }
    
    def get_daily_limit(self, subscription_tier: str) -> int:
        """Get daily generation limit for a subscription tier."""
        return self.daily_limits.get(subscription_tier.lower(), 3)
    
    def can_generate(self, user_id: int, subscription_tier: str) -> bool:
        """Check if user can generate more designs today."""
        daily_limit = self.get_daily_limit(subscription_tier)
        
        # Unlimited generations for certain tiers
        if daily_limit < 0:
            return True
        
        # Get current usage
        current_usage = self.get_current_usage(user_id)
        
        return current_usage < daily_limit
    
    def get_current_usage(self, user_id: int) -> int:
        """Get current day's usage count."""
        try:
            # Get today's date in ISO format (YYYY-MM-DD)
            today = time.strftime("%Y-%m-%d")
            
            # Query usage records
            # This is a placeholder - implement the actual database query
            # based on your usage tracking table
            
            # Placeholder implementation
            return 0  # Replace with actual usage count from database
            
        except Exception as e:
            logger.error(f"Error getting usage count: {str(e)}")
            return 0  # Default to 0 to allow generation on error
    
    def record_generation(self, user_id: int) -> None:
        """Record a generation for the user."""
        try:
            # Record the generation in the database
            # This is a placeholder - implement the actual database update
            # based on your usage tracking table
            
            # Placeholder implementation - no operation
            pass
            
        except Exception as e:
            logger.error(f"Error recording generation: {str(e)}")
    
    def get_reset_time(self) -> str:
        """Get the time when daily limits reset."""
        # Daily limits typically reset at midnight UTC
        # Calculate time until midnight UTC
        now = time.time()
        tomorrow = time.strftime("%Y-%m-%d", time.gmtime(now + 86400))
        reset_time = f"{tomorrow}T00:00:00Z"
        
        return reset_time


async def cleanup_temp_file(file_path: Path, delay_hours: int = 24) -> None:
    """
    Clean up temporary file after a delay.
    
    Args:
        file_path: Path to the file
        delay_hours: Delay in hours before deletion
    """
    try:
        # Wait for the specified delay
        await asyncio.sleep(delay_hours * 3600)
        
        # Check if file exists and delete it
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
            
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {file_path}: {str(e)}")