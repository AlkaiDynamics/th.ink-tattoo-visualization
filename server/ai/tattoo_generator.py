from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Any
import os
from datetime import datetime
from uuid import uuid4

from ..database import models, schemas, dependencies
from ..auth.security import get_current_active_user
from ..ai.model_handler import generate_tattoo_image  # Assuming this function exists

router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)

class TattooGeneratorRequest(BaseModel):
    description: str

class TattooGeneratorResponse(BaseModel):
    image_url: str

@router.post("/tattoo-generator", response_model=TattooGeneratorResponse)
def generate_tattoo(request: TattooGeneratorRequest, db: Session = Depends(dependencies.get_db), current_user: schemas.UserRead = Depends(get_current_active_user)):
    """
    Generate a tattoo image based on the provided description.
    """
    try:
        # Generate the tattoo image using the AI model
        image_bytes = generate_tattoo_image(request.description)
        
        # Generate a unique filename
        filename = f"{uuid4()}.png"
        image_path = os.path.join("static", "tattoos", filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save the image to the filesystem
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
        
        # Construct the image URL
        image_url = f"http://localhost:8000/static/tattoos/{filename}"
        
        return {"image_url": image_url}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate tattoo image") from e