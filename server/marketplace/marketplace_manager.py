"""
Marketplace management module for the Th.ink AR Tattoo Visualizer.

This module handles tattoo design listings, purchases, and artist interactions
in the marketplace, with support for searching, filtering, and recommendations.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc, and_, or_
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import json

from ..database import models, schemas, dependencies
from ..auth.security import get_current_active_user
from ..payment.payment_manager import create_payment_intent, process_payment
from ..config.model_config import get_config

# Configure logger
logger = logging.getLogger("think.marketplace")

# Initialize router
router = APIRouter(
    prefix="/marketplace",
    tags=["Marketplace"]
)

# Get configuration
config = get_config()

# Setup upload directory
UPLOAD_DIR = Path("static/uploads/marketplace")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/designs", response_model=schemas.TattooDesignRead, status_code=status.HTTP_201_CREATED)
async def create_tattoo_design(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    price: float = Form(...),
    style: str = Form(...),
    tags: List[str] = Form([]),
    is_public: bool = Form(True),
    file: UploadFile = File(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Create a new tattoo design for the marketplace.
    
    This endpoint allows artists to upload and list their designs.
    """
    try:
        # Validate the design name
        if db.query(models.TattooDesign).filter(models.TattooDesign.name == name).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A design with this name already exists"
            )
        
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PNG and JPEG images are supported"
            )
        
        # Read file
        contents = await file.read()
        
        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds the 10MB limit"
            )
        
        # Process image and create thumbnail
        try:
            image = Image.open(BytesIO(contents))
            width, height = image.size
            
            # Create design ID
            design_id = str(uuid.uuid4())
            
            # Save original image
            image_path = UPLOAD_DIR / f"{design_id}.png"
            
            # Convert to RGBA if needed
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Save original
            image.save(image_path, format="PNG")
            
            # Create thumbnail
            thumbnail_size = (256, 256)
            thumbnail = image.copy()
            thumbnail.thumbnail(thumbnail_size)
            
            # Save thumbnail
            thumbnail_path = UPLOAD_DIR / f"{design_id}_thumb.png"
            thumbnail.save(thumbnail_path, format="PNG")
            
            # Create URLs
            image_url = f"/static/uploads/marketplace/{design_id}.png"
            thumbnail_url = f"/static/uploads/marketplace/{design_id}_thumb.png"
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        # Get primary colors from image
        colors = extract_colors(image, num_colors=5)
        
        # Create the design in database
        db_design = models.TattooDesign(
            name=name,
            description=description,
            price=price,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            style=style,
            artist_id=current_user.id,
            is_public=is_public,
            width=width,
            height=height,
            is_ai_generated=False,
            colors=colors
        )
        
        db.add(db_design)
        db.commit()
        db.refresh(db_design)
        
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
                db_design.tags.append(tag)
            
            db.commit()
            db.refresh(db_design)
        
        # Log the creation
        logger.info(f"User {current_user.id} created design {db_design.id}: {name}")
        
        return db_design
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating design: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create design"
        )


@router.get("/designs", response_model=schemas.TattooDesignList)
async def list_tattoo_designs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    style: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    sort_by: str = Query("created_at", regex="^(created_at|price|name|rating)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    tags: Optional[List[str]] = Query(None),
    search: Optional[str] = Query(None),
    artist_id: Optional[int] = Query(None),
    db: Session = Depends(dependencies.get_db),
    current_user: Optional[schemas.UserRead] = Depends(get_current_active_user)
):
    """
    List all available tattoo designs with filtering and sorting.
    
    This endpoint retrieves designs from the marketplace with various filters.
    """
    try:
        # Start with base query
        query = db.query(models.TattooDesign)
        
        # Filter public designs (or all designs if viewing own designs)
        if artist_id is not None and artist_id == current_user.id:
            # Viewing own designs - include private ones
            query = query.filter(models.TattooDesign.artist_id == artist_id)
        else:
            # Viewing marketplace - only public designs
            if artist_id is not None:
                # Public designs from specific artist
                query = query.filter(
                    models.TattooDesign.is_public == True,
                    models.TattooDesign.artist_id == artist_id
                )
            else:
                # All public designs
                query = query.filter(models.TattooDesign.is_public == True)
        
        # Apply filters
        if style:
            query = query.filter(models.TattooDesign.style == style)
            
        if min_price is not None:
            query = query.filter(models.TattooDesign.price >= min_price)
            
        if max_price is not None:
            query = query.filter(models.TattooDesign.price <= max_price)
            
        if tags:
            # Filter designs that have all the specified tags
            for tag in tags:
                tag_query = db.query(models.Tag.id).filter(models.Tag.name == tag).scalar_subquery()
                query = query.filter(
                    models.TattooDesign.tags.any(models.Tag.id == tag_query)
                )
        
        if search:
            # Search in name and description
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    models.TattooDesign.name.ilike(search_term),
                    models.TattooDesign.description.ilike(search_term)
                )
            )
        
        # Get total count for pagination
        total = query.count()
        
        # Apply sorting
        if sort_order.lower() == "asc":
            sort_func = asc
        else:
            sort_func = desc
            
        if sort_by == "created_at":
            query = query.order_by(sort_func(models.TattooDesign.created_at))
        elif sort_by == "price":
            query = query.order_by(sort_func(models.TattooDesign.price))
        elif sort_by == "name":
            query = query.order_by(sort_func(models.TattooDesign.name))
        elif sort_by == "rating":
            query = query.order_by(sort_func(models.TattooDesign.rating))
        
        # Apply pagination
        designs = query.offset(skip).limit(limit).all()
        
        # Calculate pagination info
        page = skip // limit + 1
        total_pages = (total + limit - 1) // limit  # Ceiling division
        
        return {
            "designs": designs,
            "total": total,
            "page": page,
            "per_page": limit,
            "pages": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error listing designs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list designs"
        )


@router.get("/designs/{design_id}", response_model=schemas.TattooDesignRead)
async def get_tattoo_design(
    design_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: Optional[schemas.UserRead] = Depends(get_current_active_user)
):
    """
    Retrieve a specific tattoo design by ID.
    
    This endpoint gets detailed information about a specific design.
    """
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Check if user has access to this design
    if not design.is_public and (not current_user or design.artist_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this design"
        )
    
    # Increment view count
    design.view_count += 1
    db.commit()
    
    return design


@router.put("/designs/{design_id}", response_model=schemas.TattooDesignRead)
async def update_tattoo_design(
    design_id: int = Path(...),
    design_update: schemas.TattooDesignUpdate = Body(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Update an existing tattoo design.
    
    This endpoint allows artists to update their design information.
    """
    # Get the design
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Check ownership
    if design.artist_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to update this design"
        )
    
    # Update fields if provided
    if design_update.name is not None:
        # Check if name is already taken
        existing = db.query(models.TattooDesign).filter(
            models.TattooDesign.name == design_update.name,
            models.TattooDesign.id != design_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A design with this name already exists"
            )
        
        design.name = design_update.name
        
    if design_update.description is not None:
        design.description = design_update.description
        
    if design_update.price is not None:
        design.price = design_update.price
        
    if design_update.style is not None:
        design.style = design_update.style
        
    if design_update.is_public is not None:
        design.is_public = design_update.is_public
    
    # Update tags if provided
    if design_update.tags is not None:
        # Clear existing tags
        design.tags = []
        
        # Add new tags
        for tag_name in design_update.tags:
            # Find or create tag
            tag = db.query(models.Tag).filter(models.Tag.name == tag_name).first()
            if not tag:
                tag = models.Tag(name=tag_name)
                db.add(tag)
                db.commit()
                db.refresh(tag)
            
            # Add association
            design.tags.append(tag)
    
    # Update timestamp
    design.updated_at = datetime.utcnow()
    
    # Save changes
    db.commit()
    db.refresh(design)
    
    return design


@router.delete("/designs/{design_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tattoo_design(
    design_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Delete a tattoo design.
    
    This endpoint allows artists to remove their designs from the marketplace.
    """
    # Get the design
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Check ownership
    if design.artist_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this design"
        )
    
    # Check if design has been purchased
    if design.purchase_count > 0:
        # Soft delete instead of hard delete
        design.is_public = False
        design.updated_at = datetime.utcnow()
        db.commit()
    else:
        # Hard delete
        # Delete image files
        try:
            # Extract filename from URL
            filename = os.path.basename(design.image_url)
            thumbnail_filename = os.path.basename(design.thumbnail_url)
            
            # Delete files
            image_path = UPLOAD_DIR / filename
            thumbnail_path = UPLOAD_DIR / thumbnail_filename
            
            if image_path.exists():
                image_path.unlink()
            
            if thumbnail_path.exists():
                thumbnail_path.unlink()
                
        except Exception as e:
            logger.error(f"Error deleting design files: {str(e)}")
        
        # Delete from database
        db.delete(design)
        db.commit()
    
    return None


@router.post("/designs/{design_id}/purchase", response_model=schemas.PaymentRead)
async def purchase_design(
    design_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Purchase a tattoo design.
    
    This endpoint initiates the payment process for a design purchase.
    """
    # Get the design
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Check if design is public and available
    if not design.is_public:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This design is not available for purchase"
        )
    
    # Check if user already owns this design
    # (would need a purchases or owned_designs table to track this)
    
    # Create payment
    try:
        # Create a payment intent
        payment = await create_payment_intent(
            db=db,
            user_id=current_user.id,
            amount=design.price,
            currency="USD",
            design_id=design.id
        )
        
        return payment
        
    except Exception as e:
        logger.error(f"Error creating payment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate payment"
        )


@router.post("/designs/{design_id}/review", response_model=schemas.DesignReviewRead)
async def review_design(
    design_id: int = Path(...),
    review: schemas.DesignReviewCreate = Body(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Add a review for a tattoo design.
    
    This endpoint allows users to rate and review designs they've purchased.
    """
    # Get the design
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Check if user has already reviewed this design
    existing_review = db.query(models.DesignReview).filter(
        models.DesignReview.design_id == design_id,
        models.DesignReview.user_id == current_user.id
    ).first()
    
    if existing_review:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already reviewed this design"
        )
    
    # Create the review
    db_review = models.DesignReview(
        design_id=design_id,
        user_id=current_user.id,
        rating=review.rating,
        review_text=review.review_text
    )
    
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    
    # Update design rating
    reviews = db.query(models.DesignReview).filter(models.DesignReview.design_id == design_id).all()
    total_rating = sum(r.rating for r in reviews)
    avg_rating = total_rating / len(reviews)
    
    design.rating = avg_rating
    design.rating_count = len(reviews)
    db.commit()
    
    return db_review


@router.get("/tags", response_model=List[Dict[str, Any]])
async def get_tags(
    db: Session = Depends(dependencies.get_db)
):
    """
    Get all available tags for filtering designs.
    
    This endpoint returns all tags with their usage counts.
    """
    # Query all tags with their usage count
    tags = db.query(
        models.Tag.id,
        models.Tag.name,
        func.count(models.design_tags.c.design_id).label("count")
    ).outerjoin(
        models.design_tags
    ).group_by(
        models.Tag.id
    ).order_by(
        desc("count")
    ).all()
    
    # Format response
    return [{"id": tag.id, "name": tag.name, "count": tag.count} for tag in tags]


@router.get("/styles", response_model=List[Dict[str, Any]])
async def get_styles(
    db: Session = Depends(dependencies.get_db)
):
    """
    Get all available tattoo styles with their usage counts.
    
    This endpoint returns all styles for filtering designs.
    """
    # Query all styles with their usage count
    styles = db.query(
        models.TattooDesign.style,
        func.count(models.TattooDesign.id).label("count")
    ).filter(
        models.TattooDesign.is_public == True
    ).group_by(
        models.TattooDesign.style
    ).order_by(
        desc("count")
    ).all()
    
    # Style descriptions
    style_descriptions = {
        "traditional": "Bold black outlines with limited color palette",
        "realistic": "Photorealistic detail with shading and depth",
        "blackwork": "Solid black ink with intricate patterns",
        "watercolor": "Soft color blending with paint splatter effects",
        "tribal": "Bold black tribal patterns with cultural influences",
        "japanese": "Traditional Japanese iconography and techniques",
        "new_school": "Cartoonish style with exaggerated proportions and vibrant colors",
        "minimalist": "Clean, simple lines with minimal detail",
        "geometric": "Precise geometric shapes and patterns",
        "dotwork": "Intricate patterns of tiny dots creating detailed shading"
    }
    
    # Format response
    return [{
        "id": style.style,
        "name": style.style.capitalize(),
        "count": style.count,
        "description": style_descriptions.get(style.style, "")
    } for style in styles]


@router.get("/recommended", response_model=List[schemas.TattooDesignRead])
async def get_recommended_designs(
    limit: int = Query(6, ge=1, le=20),
    db: Session = Depends(dependencies.get_db),
    current_user: Optional[schemas.UserRead] = Depends(get_current_active_user)
):
    """
    Get personalized design recommendations.
    
    This endpoint returns designs based on user preferences and popular items.
    """
    try:
        # Base query for public designs
        query = db.query(models.TattooDesign).filter(models.TattooDesign.is_public == True)
        
        if current_user:
            # Personalized recommendations if user is logged in
            
            # Get user's purchase/view history
            # This is a simplified implementation - in a real system, you'd use a more
            # sophisticated recommendation engine based on collaborative filtering or similar
            
            # Get user's favorite styles (from purchases or view history)
            user_styles = db.query(models.TattooDesign.style).filter(
                models.TattooDesign.favorited_by.any(models.User.id == current_user.id)
            ).group_by(models.TattooDesign.style).limit(3).all()
            
            user_styles = [style[0] for style in user_styles]
            
            if user_styles:
                # Recommend designs with similar styles
                recommended = query.filter(models.TattooDesign.style.in_(user_styles))
                
                # Exclude designs the user already favorited
                recommended = recommended.filter(
                    ~models.TattooDesign.favorited_by.any(models.User.id == current_user.id)
                )
                
                # Order by rating and recency
                recommended = recommended.order_by(
                    desc(models.TattooDesign.rating),
                    desc(models.TattooDesign.created_at)
                )
                
                # Get results
                designs = recommended.limit(limit).all()
                
                if len(designs) >= limit // 2:
                    return designs[:limit]
        
        # Fallback to popular designs if not enough personalized recommendations
        popular = query.order_by(
            desc(models.TattooDesign.rating),
            desc(models.TattooDesign.purchase_count),
            desc(models.TattooDesign.view_count)
        ).limit(limit).all()
        
        return popular
        
    except Exception as e:
        logger.error(f"Error getting recommended designs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recommendations"
        )


@router.post("/designs/{design_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
async def favorite_design(
    design_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Add a design to user's favorites.
    
    This endpoint allows users to save designs to their favorites.
    """
    # Get the design
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Check if design is public
    if not design.is_public:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This design is not available"
        )
    
    # Get the user
    user = db.query(models.User).filter(models.User.id == current_user.id).first()
    
    # Add to favorites if not already there
    if design not in user.favorite_designs:
        user.favorite_designs.append(design)
        db.commit()
    
    return None


@router.delete("/designs/{design_id}/favorite", status_code=status.HTTP_204_NO_CONTENT)
async def unfavorite_design(
    design_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Remove a design from user's favorites.
    
    This endpoint allows users to remove designs from their favorites.
    """
    # Get the design
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    
    if not design:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tattoo design not found"
        )
    
    # Get the user
    user = db.query(models.User).filter(models.User.id == current_user.id).first()
    
    # Remove from favorites if present
    if design in user.favorite_designs:
        user.favorite_designs.remove(design)
        db.commit()
    
    return None


@router.get("/favorites", response_model=schemas.TattooDesignList)
async def get_favorites(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Get user's favorited designs.
    
    This endpoint returns all designs a user has marked as favorites.
    """
    try:
        # Get user with favorites
        user = db.query(models.User).filter(models.User.id == current_user.id).first()
        
        # Get total count
        total = len(user.favorite_designs)
        
        # Apply pagination (basic implementation)
        favorites = user.favorite_designs[skip:skip+limit]
        
        # Calculate pagination info
        page = skip // limit + 1
        total_pages = (total + limit - 1) // limit  # Ceiling division
        
        return {
            "designs": favorites,
            "total": total,
            "page": page,
            "per_page": limit,
            "pages": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error getting favorites: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get favorites"
        )


# Helper functions

def extract_colors(image: Image.Image, num_colors: int = 5) -> List[str]:
    """
    Extract dominant colors from an image.
    
    Args:
        image: PIL Image object
        num_colors: Number of colors to extract
        
    Returns:
        List of hex color codes
    """
    try:
        # Resize image for faster processing
        img_small = image.copy()
        img_small.thumbnail((100, 100))
        
        # Convert to RGB if needed
        if img_small.mode != 'RGB':
            img_small = img_small.convert('RGB')
        
        # Get pixel data
        pixels = list(img_small.getdata())
        
        # Count color occurrences
        color_count = {}
        for pixel in pixels:
            color = pixel[:3]  # RGB values
            if color in color_count:
                color_count[color] += 1
            else:
                color_count[color] = 1
        
        # Sort by occurrence count
        sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N colors and convert to hex
        top_colors = []
        for color, _ in sorted_colors[:num_colors]:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            top_colors.append(hex_color)
        
        return top_colors
        
    except Exception as e:
        logger.error(f"Error extracting colors: {str(e)}")
        return []  # Return empty list on error