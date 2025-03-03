from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..database import models, schemas, dependencies

router = APIRouter(
    prefix="/subscriptions",
    tags=["Subscriptions"]
)

@router.post("/", response_model=schemas.SubscriptionRead)
def create_subscription(subscription: schemas.SubscriptionCreate, db: Session = Depends(dependencies.get_db)):
    # Check if user exists
    user = db.query(models.User).filter(models.User.id == subscription.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if subscription type is valid
    if subscription.subscription_type not in ["Free", "Premium", "Pro"]:
        raise HTTPException(status_code=400, detail="Invalid subscription type")
    
    db_subscription = models.Subscription(
        user_id=subscription.user_id,
        subscription_type=subscription.subscription_type,
        status="active",
        start_date=subscription.start_date,
        end_date=subscription.end_date
    )
    db.add(db_subscription)
    db.commit()
    db.refresh(db_subscription)
    return db_subscription

@router.get("/", response_model=List[schemas.SubscriptionRead])
def read_subscriptions(skip: int = 0, limit: int = 10, db: Session = Depends(dependencies.get_db)):
    subscriptions = db.query(models.Subscription).offset(skip).limit(limit).all()
    return subscriptions

@router.get("/{subscription_id}", response_model=schemas.SubscriptionRead)
def read_subscription(subscription_id: int, db: Session = Depends(dependencies.get_db)):
    subscription = db.query(models.Subscription).filter(models.Subscription.id == subscription_id).first()
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return subscription

@router.put("/{subscription_id}", response_model=schemas.SubscriptionRead)
def update_subscription(subscription_id: int, subscription: schemas.SubscriptionUpdate, db: Session = Depends(dependencies.get_db)):
    db_subscription = db.query(models.Subscription).filter(models.Subscription.id == subscription_id).first()
    if not db_subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    if subscription.subscription_type:
        if subscription.subscription_type not in ["Free", "Premium", "Pro"]:
            raise HTTPException(status_code=400, detail="Invalid subscription type")
        db_subscription.subscription_type = subscription.subscription_type
    
    if subscription.status:
        db_subscription.status = subscription.status
    
    if subscription.end_date:
        db_subscription.end_date = subscription.end_date
    
    db.commit()
    db.refresh(db_subscription)
    return db_subscription

@router.delete("/{subscription_id}", response_model=schemas.SubscriptionRead)
def delete_subscription(subscription_id: int, db: Session = Depends(dependencies.get_db)):
    subscription = db.query(models.Subscription).filter(models.Subscription.id == subscription_id).first()
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    db.delete(subscription)
    db.commit()
    return subscription