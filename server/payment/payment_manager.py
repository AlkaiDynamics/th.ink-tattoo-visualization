"""
Payment processing module for the Th.ink AR Tattoo Visualizer.

This module provides functionality for creating and managing payments, 
integrating with payment processors like Stripe, and handling subscriptions
and purchases in the tattoo marketplace.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks, Body, Path, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import stripe
import os
import logging
import json
from uuid import uuid4

from ..database import models, schemas, dependencies
from ..auth.security import get_current_active_user
from ..config.model_config import get_config
from ..errors.error_handler import handle_errors

# Configure logger
logger = logging.getLogger("think.payment")

# Initialize router
router = APIRouter(
    prefix="/payments",
    tags=["Payments"]
)

# Get configuration
config = get_config()

# Initialize Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
stripe_webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


@router.post("/create-payment-intent", response_model=schemas.PaymentRead)
async def create_payment(
    payment: schemas.PaymentCreate,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Create a payment intent for a purchase.
    
    This endpoint initializes a payment and returns the client secret for processing.
    """
    try:
        # If no payment details provided
        if not payment.design_id and not payment.subscription_tier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either design_id or subscription_tier must be provided"
            )
        
        # Create a Stripe Payment Intent
        payment_intent = await create_payment_intent(
            db=db,
            user_id=current_user.id,
            amount=payment.amount,
            currency=payment.currency,
            design_id=payment.design_id,
            subscription_tier=payment.subscription_tier
        )
        
        return payment_intent
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment processing error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Payment creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment"
        )


@router.get("/", response_model=schemas.PaymentList)
async def get_payments(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Retrieve a list of payments for the authenticated user.
    
    This endpoint returns payment history for the current user.
    """
    try:
        # Build query
        query = db.query(models.Payment).filter(models.Payment.user_id == current_user.id)
        
        # Filter by status if provided
        if status:
            query = query.filter(models.Payment.status == status)
        
        # Get total count for pagination
        total = query.count()
        
        # Apply pagination
        payments = query.order_by(models.Payment.created_at.desc()).offset(skip).limit(limit).all()
        
        # Return results
        return {
            "payments": payments,
            "total": total,
            "page": skip // limit + 1,
            "per_page": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting payments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve payments"
        )


@router.get("/{payment_id}", response_model=schemas.PaymentRead)
async def get_payment(
    payment_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Retrieve details of a specific payment.
    
    This endpoint gets detailed information about a payment.
    """
    payment = db.query(models.Payment).filter(
        models.Payment.id == payment_id,
        models.Payment.user_id == current_user.id
    ).first()
    
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found"
        )
    
    return payment


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(dependencies.get_db)
):
    """
    Handle Stripe webhook events to update payment statuses.
    
    This endpoint receives webhooks from Stripe and processes payment events.
    """
    try:
        # Get the webhook payload
        payload = await request.body()
        signature = request.headers.get("stripe-signature", "")
        
        # Verify the webhook signature
        try:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature,
                secret=stripe_webhook_secret
            )
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid signature"
            )
        
        # Process the event
        event_data = event.data.object
        
        if event.type == "payment_intent.succeeded":
            # Payment succeeded - process the successful payment
            background_tasks.add_task(
                process_successful_payment,
                db=db,
                payment_intent_id=event_data.id
            )
            
        elif event.type == "payment_intent.payment_failed":
            # Payment failed - update the payment status
            background_tasks.add_task(
                process_failed_payment,
                db=db,
                payment_intent_id=event_data.id,
                failure_message=event_data.get("last_payment_error", {}).get("message", "")
            )
            
        # Return success response
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook"
        )


@router.post("/confirm-payment/{payment_id}", response_model=schemas.PaymentRead)
async def confirm_payment(
    payment_id: int = Path(...),
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Manually confirm a payment (for testing purposes).
    
    This endpoint allows manual confirmation of payments in non-production environments.
    """
    # Check if in development mode
    if config.env != "development":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only available in development mode"
        )
    
    # Get the payment
    payment = db.query(models.Payment).filter(
        models.Payment.id == payment_id,
        models.Payment.user_id == current_user.id
    ).first()
    
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found"
        )
    
    # Process the payment
    await process_payment(db=db, payment=payment)
    
    # Refresh and return the payment
    db.refresh(payment)
    return payment


# Helper functions

@handle_errors()
async def create_payment_intent(
    db: Session,
    user_id: int,
    amount: float,
    currency: str,
    design_id: Optional[int] = None,
    subscription_tier: Optional[str] = None
) -> models.Payment:
    """
    Create a Stripe Payment Intent and record in database.
    
    Args:
        db: Database session
        user_id: User ID
        amount: Payment amount
        currency: Currency code
        design_id: Design ID if purchasing a design
        subscription_tier: Subscription tier if purchasing a subscription
        
    Returns:
        Payment record from database
    """
    try:
        # Validate the payment amount
        if amount <= 0:
            raise ValueError("Payment amount must be positive")
        
        # Convert amount to cents for Stripe
        stripe_amount = int(amount * 100)
        
        # Create metadata for the payment
        metadata = {
            "user_id": str(user_id)
        }
        
        if design_id is not None:
            metadata["design_id"] = str(design_id)
            
            # Get design details for payment description
            design = db.query(models.TattooDesign).filter(models.TattooDesign.id == design_id).first()
            description = f"Purchase of design: {design.name}" if design else "Design purchase"
        
        elif subscription_tier is not None:
            metadata["subscription_tier"] = subscription_tier
            description = f"Subscription: {subscription_tier.capitalize()}"
        
        else:
            description = "Payment to Th.ink AR"
        
        # Create the Stripe Payment Intent
        intent = stripe.PaymentIntent.create(
            amount=stripe_amount,
            currency=currency.lower(),
            metadata=metadata,
            description=description,
            payment_method_types=["card"],
            receipt_email=db.query(models.User.email).filter(models.User.id == user_id).scalar()
        )
        
        # Create database record
        payment = models.Payment(
            user_id=user_id,
            amount=amount,
            currency=currency.upper(),
            status="pending",
            external_payment_id=intent.id,
            design_id=design_id,
            payment_method="card",
            metadata=metadata
        )
        
        # If for subscription, create subscription record
        if subscription_tier is not None:
            # Calculate subscription duration (1 month by default)
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=30)
            
            # Create subscription
            subscription = models.Subscription(
                user_id=user_id,
                subscription_tier=subscription_tier,
                status="pending",
                start_date=start_date,
                end_date=end_date,
                auto_renew=True
            )
            
            db.add(subscription)
            db.flush()  # Get the ID without committing
            
            # Link payment to subscription
            payment.subscription_id = subscription.id
        
        # Save the payment
        db.add(payment)
        db.commit()
        db.refresh(payment)
        
        # Add client_secret to metadata for client-side confirmation
        payment.metadata = {
            **payment.metadata if payment.metadata else {},
            "client_secret": intent.client_secret
        }
        
        return payment
        
    except Exception as e:
        logger.error(f"Error creating payment intent: {str(e)}")
        raise


async def process_successful_payment(db: Session, payment_intent_id: str) -> None:
    """
    Process a successful payment from webhook.
    
    Args:
        db: Database session
        payment_intent_id: Stripe Payment Intent ID
    """
    try:
        # Get the payment
        payment = db.query(models.Payment).filter(
            models.Payment.external_payment_id == payment_intent_id
        ).first()
        
        if not payment:
            logger.error(f"Payment not found for intent ID: {payment_intent_id}")
            return
        
        # Process the payment
        await process_payment(db, payment)
        
    except Exception as e:
        logger.error(f"Error processing successful payment: {str(e)}")


async def process_failed_payment(
    db: Session,
    payment_intent_id: str,
    failure_message: str
) -> None:
    """
    Process a failed payment from webhook.
    
    Args:
        db: Database session
        payment_intent_id: Stripe Payment Intent ID
        failure_message: Error message from payment processor
    """
    try:
        # Get the payment
        payment = db.query(models.Payment).filter(
            models.Payment.external_payment_id == payment_intent_id
        ).first()
        
        if not payment:
            logger.error(f"Payment not found for intent ID: {payment_intent_id}")
            return
        
        # Update payment status
        payment.status = "failed"
        payment.updated_at = datetime.utcnow()
        payment.metadata = {
            **payment.metadata if payment.metadata else {},
            "failure_message": failure_message
        }
        
        # Handle subscription if applicable
        if payment.subscription_id:
            subscription = db.query(models.Subscription).filter(
                models.Subscription.id == payment.subscription_id
            ).first()
            
            if subscription:
                subscription.status = "failed"
                subscription.updated_at = datetime.utcnow()
        
        # Save changes
        db.commit()
        
        logger.info(f"Payment {payment.id} marked as failed")
        
    except Exception as e:
        logger.error(f"Error processing failed payment: {str(e)}")


async def process_payment(db: Session, payment: models.Payment) -> None:
    """
    Process a successful payment.
    
    Args:
        db: Database session
        payment: Payment model instance
    """
    try:
        # Update payment status
        payment.status = "completed"
        payment.completed_at = datetime.utcnow()
        payment.updated_at = datetime.utcnow()
        
        # Handle design purchase
        if payment.design_id:
            design = db.query(models.TattooDesign).filter(
                models.TattooDesign.id == payment.design_id
            ).first()
            
            if design:
                # Increment purchase count
                design.purchase_count += 1
                
                # Add to user's purchased designs
                # (This would require a separate table or relationship)
                
                # Add commission to artist's account
                artist_commission = payment.amount * 0.7  # 70% to artist
                
                # Record commission transaction
                # (This would require a separate table for artist earnings)
        
        # Handle subscription payment
        if payment.subscription_id:
            subscription = db.query(models.Subscription).filter(
                models.Subscription.id == payment.subscription_id
            ).first()
            
            if subscription:
                # Activate subscription
                subscription.status = "active"
                subscription.updated_at = datetime.utcnow()
                
                # Update user's subscription tier
                user = db.query(models.User).filter(
                    models.User.id == payment.user_id
                ).first()
                
                if user:
                    user.subscription_tier = subscription.subscription_tier
        
        # Save changes
        db.commit()
        
        logger.info(f"Payment {payment.id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing payment: {str(e)}")
        db.rollback()
        raise