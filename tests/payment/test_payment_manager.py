import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import stripe
from unittest.mock import patch
from datetime import datetime

from server.app import app
from server.database.models import Base, User, TattooDesign, Payment
from server.database.dependencies import get_db
from server.auth.security import get_password_hash

# Create a new SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_payments.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency override for testing
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="module")
def test_user():
    hashed_password = get_password_hash("testpassword")
    user = User(username="testuser", email="testuser@example.com", full_name="Test User", hashed_password=hashed_password)
    return user

@pytest.fixture(scope="module")
def test_design(test_user):
    design = TattooDesign(name="Test Design", description="A test tattoo design.", price=50.0, artist_id=test_user.id)
    return design

@pytest.fixture(scope="module")
def setup_db(test_user, test_design):
    db = TestingSessionLocal()
    db.add(test_user)
    db.add(test_design)
    db.commit()
    db.refresh(test_user)
    db.refresh(test_design)
    yield
    db.close()

def authenticate_user():
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    return response.json()["access_token"]

@patch('stripe.PaymentIntent.create')
def test_create_payment_intent_success(mock_stripe_create, setup_db, test_user):
    token = authenticate_user()
    mock_stripe_create.return_value = stripe.PaymentIntent.construct_from({
        "id": "pi_test",
        "object": "payment_intent",
        "amount": 5000,
        "currency": "usd",
        "metadata": {
            "integration_check": "accept_a_payment",
            "user_id": test_user.id
        }
    }, stripe.api_key)
    
    payment_data = {
        "amount": 50.0,
        "currency": "USD"
    }
    response = client.post("/payments/create-payment-intent", json=payment_data, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["amount"] == 50.0
    assert data["currency"] == "usd"
    assert data["status"] == "requires_payment_method"
    assert data["external_payment_id"] == "pi_test"
    assert data["user_id"] == test_user.id

@patch('stripe.PaymentIntent.create')
def test_create_payment_intent_failure(mock_stripe_create, setup_db, test_user):
    token = authenticate_user()
    mock_stripe_create.side_effect = stripe.error.StripeError("Stripe error")
    
    payment_data = {
        "amount": 50.0,
        "currency": "USD"
    }
    response = client.post("/payments/create-payment-intent", json=payment_data, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 400
    data = response.json()
    assert "Stripe error" in data["detail"]

def test_get_payments_empty(setup_db):
    token = authenticate_user()
    response = client.get("/payments/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

@patch('stripe.PaymentIntent.create')
def test_get_payments(setup_db, test_user, test_design, mock_stripe_create):
    token = authenticate_user()
    mock_stripe_create.return_value = stripe.PaymentIntent.construct_from({
        "id": "pi_test_2",
        "object": "payment_intent",
        "amount": 5000,
        "currency": "usd",
        "metadata": {
            "integration_check": "accept_a_payment",
            "user_id": test_user.id
        }
    }, stripe.api_key)
    
    # Create a payment
    payment_data = {
        "amount": 50.0,
        "currency": "USD"
    }
    create_response = client.post("/payments/create-payment-intent", json=payment_data, headers={"Authorization": f"Bearer {token}"})
    assert create_response.status_code == 200
    
    # Retrieve payments
    response = client.get("/payments/", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["amount"] == 50.0
    assert data[0]["currency"] == "usd"
    assert data[0]["status"] == "requires_payment_method"
    assert data[0]["external_payment_id"] == "pi_test_2"

def test_get_payment_success(setup_db, test_user, test_design):
    token = authenticate_user()
    # Create a payment directly in the database
    db = TestingSessionLocal()
    payment = Payment(user_id=test_user.id, amount=50.0, currency="usd", status="pending", external_payment_id="pi_test_3")
    db.add(payment)
    db.commit()
    db.refresh(payment)
    
    response = client.get(f"/payments/{payment.id}", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == payment.id
    assert data["amount"] == 50.0
    assert data["currency"] == "usd"
    assert data["status"] == "pending"
    assert data["external_payment_id"] == "pi_test_3"
    assert data["user_id"] == test_user.id

def test_get_payment_not_found(setup_db, test_user):
    token = authenticate_user()
    response = client.get("/payments/999", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Payment not found."

@patch('stripe.PaymentIntent.create')
def test_stripe_webhook_payment_intent_succeeded(mock_stripe_create, setup_db, test_user):
    token = authenticate_user()
    # Create a payment
    db = TestingSessionLocal()
    payment = Payment(user_id=test_user.id, amount=50.0, currency="usd", status="requires_payment_method", external_payment_id="pi_test_4")
    db.add(payment)
    db.commit()
    db.refresh(payment)
    
    payload = {
        "id": "evt_test",
        "object": "event",
        "type": "payment_intent.succeeded",
        "data": {
            "object": {
                "id": "pi_test_4",
                "object": "payment_intent",
                "amount": 5000,
                "currency": "usd",
                "metadata": {
                    "user_id": test_user.id
                }
            }
        }
    }
    sig_header = "test_signature"
    endpoint_secret = "whsec_test"

    with patch('stripe.Webhook.construct_event') as mock_construct_event:
        mock_construct_event.return_value = payload
        response = client.post("/payments/webhook", json=payload, headers={"stripe-signature": sig_header})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    # Verify that the payment status was updated
    updated_payment = db.query(Payment).filter(Payment.external_payment_id == "pi_test_4").first()
    assert updated_payment.status == "succeeded"

@patch('stripe.Webhook.construct_event')
def test_stripe_webhook_invalid_payload(mock_construct_event, setup_db):
    mock_construct_event.side_effect = ValueError("Invalid payload")
    response = client.post("/payments/webhook", data="invalid", headers={"stripe-signature": "invalid"})
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid payload."

@patch('stripe.Webhook.construct_event')
def test_stripe_webhook_invalid_signature(mock_construct_event, setup_db):
    mock_construct_event.side_effect = stripe.error.SignatureVerificationError("Invalid signature", "test_sig")
    response = client.post("/payments/webhook", data="{}", headers={"stripe-signature": "invalid"})
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid signature."