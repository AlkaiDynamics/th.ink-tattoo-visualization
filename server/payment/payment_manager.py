from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from typing import List

import stripe
import os
from datetime import datetime

from ..database import models, schemas, dependencies
from ..auth.security import get_current_active_user

router = APIRouter(
    prefix="/payments",
    tags=["Payments"]
)

# Initialize Stripe with your secret key from environment variables
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

@router.post("/create-payment-intent", response_model=schemas.PaymentRead)
def create_payment_intent(
    payment: schemas.PaymentCreate,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Create a Stripe Payment Intent for a given amount and currency.
    """
    try:
        intent = stripe.PaymentIntent.create(
            amount=int(payment.amount * 100),  # Convert dollars to cents
            currency=payment.currency.lower(),
            metadata={'integration_check': 'accept_a_payment', 'user_id': current_user.id}
        )
        
        # Create a new payment record in the database
        db_payment = models.Payment(
            user_id=current_user.id,
            amount=payment.amount,
            currency=payment.currency.lower(),
            status='requires_payment_method',
            external_payment_id=intent.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(db_payment)
        db.commit()
        db.refresh(db_payment)
        
        return db_payment
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[schemas.PaymentRead])
def get_payments(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Retrieve a list of payments for the authenticated user.
    """
    payments = db.query(models.Payment).filter(models.Payment.user_id == current_user.id).offset(skip).limit(limit).all()
    return payments


@router.get("/{payment_id}", response_model=schemas.PaymentRead)
def get_payment(
    payment_id: int,
    db: Session = Depends(dependencies.get_db),
    current_user: schemas.UserRead = Depends(get_current_active_user)
):
    """
    Retrieve details of a specific payment.
    """
    payment = db.query(models.Payment).filter(models.Payment.id == payment_id, models.Payment.user_id == current_user.id).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found.")
    return payment


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(dependencies.get_db)):
    """
    Handle Stripe webhook events to update payment statuses.
    """
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload.")
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature.")

    # Handle the event
    if event['type'] == 'payment_intent.succeeded':
        intent = event['data']['object']
        payment_id = intent['metadata'].get('payment_id')
        if payment_id:
            payment = db.query(models.Payment).filter(models.Payment.external_payment_id == intent['id']).first()
            if payment:
                payment.status = 'succeeded'
                payment.updated_at = datetime.utcnow()
                db.commit()
    elif event['type'] == 'payment_intent.payment_failed':
        intent = event['data']['object']
        payment_id = intent['metadata'].get('payment_id')
        if payment_id:
            payment = db.query(models.Payment).filter(models.Payment.external_payment_id == intent['id']).first()
            if payment:
                payment.status = 'failed'
                payment.updated_at = datetime.utcnow()
                db.commit()
    # ... handle other event types

    return {"status": "success"}