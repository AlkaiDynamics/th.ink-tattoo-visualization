from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..database import models, schemas, dependencies
from ..auth.security import get_current_active_user

router = APIRouter(
    prefix="/marketplace",
    tags=["Marketplace"]
)

@router.post("/", response_model=schemas.TattooDesignRead, status_code=status.HTTP_201_CREATED)
def create_tattoo_design(
    design: schemas.TattooDesignCreate,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Create a new tattoo design. Only authenticated users can create designs.
    """
    # Check if design name already exists
    existing_design = db.query(models.TattooDesign).filter(models.TattooDesign.name == design.name).first()
    if existing_design:
        raise HTTPException(status_code=400, detail="Tattoo design with this name already exists.")
    
    db_design = models.TattooDesign(
        name=design.name,
        description=design.description,
        price=design.price,
        artist_id=current_user.id
    )
    db.add(db_design)
    db.commit()
    db.refresh(db_design)
    return db_design

@router.get("/", response_model=List[schemas.TattooDesignRead])
def list_tattoo_designs(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(dependencies.get_db)
):
    """
    List all available tattoo designs.
    """
    designs = db.query(models.TattooDesign).offset(skip).limit(limit).all()
    return designs

@router.get("/{design_id}", response_model=schemas.TattooDesignRead)
def get_tattoo_design(
    design_id: int,
    db: Session = Depends(dependencies.get_db)
):
    """
    Retrieve a specific tattoo design by ID.
    """
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    if not design:
        raise HTTPException(status_code=404, detail="Tattoo design not found.")
    return design

@router.put("/{design_id}", response_model=schemas.TattooDesignRead)
def update_tattoo_design(
    design_id: int,
    design_update: schemas.TattooDesignUpdate,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Update an existing tattoo design. Only the artist who created the design can update it.
    """
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    if not design:
        raise HTTPException(status_code=404, detail="Tattoo design not found.")
    if design.artist_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this design.")
    
    if design_update.name:
        # Check if new name is unique
        existing_design = db.query(models.TattooDesign).filter(models.TattooDesign.name == design_update.name).first()
        if existing_design and existing_design.id != design_id:
            raise HTTPException(status_code=400, detail="Another design with this name already exists.")
        design.name = design_update.name
    if design_update.description is not None:
        design.description = design_update.description
    if design_update.price is not None:
        design.price = design_update.price
    
    db.commit()
    db.refresh(design)
    return design

@router.delete("/{design_id}", response_model=schemas.TattooDesignRead)
def delete_tattoo_design(
    design_id: int,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Delete a tattoo design. Only the artist who created the design can delete it.
    """
    design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
    if not design:
        raise HTTPException(status_code=404, detail="Tattoo design not found.")
    if design.artist_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this design.")
    
    db.delete(design)
    db.commit()
    return design