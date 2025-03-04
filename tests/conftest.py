"""
Test configuration and fixtures for the Th.ink AR project.
"""

import os
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from server.app import app
from server.database.models import Base
from server.database.dependencies import get_db
from src.config.model_config import ModelConfig


# Database fixture
@pytest.fixture(scope="session")
def db_engine():
    """Create a SQLite in-memory database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Get a SQLAlchemy session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def client(db_session):
    """Get a test client for FastAPI."""
    # Override the get_db dependency
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def auth_client(client):
    """Get a test client with authentication."""
    # Log in and get token
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    
    # Create test user if doesn't exist
    from server.auth.security import get_password_hash
    from server.database.models import User
    db = next(app.dependency_overrides[get_db]())
    if not db.query(User).filter(User.username == "testuser").first():
        test_user = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password=get_password_hash("testpassword")
        )
        db.add(test_user)
        db.commit()
    
    response = client.post("/auth/login", json=login_data)
    token = response.json()["access_token"]
    client.headers = {
        "Authorization": f"Bearer {token}",
        **client.headers
    }
    
    return client


@pytest.fixture(scope="session")
def test_config():
    """Get test configuration."""
    return ModelConfig()


@pytest.fixture(scope="session")
def test_image():
    """Get a test image for AR testing."""
    import numpy as np
    
    # Create a simple test image (100x100 RGB)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some content to the image (a red square)
    image[25:75, 25:75, 0] = 255
    
    return image


# Mock fixtures for external services
@pytest.fixture(scope="function")
def mock_stripe(monkeypatch):
    """Mock Stripe API responses."""
    class MockStripe:
        @staticmethod
        def PaymentIntent(*args, **kwargs):
            return type('obj', (object,), {
                'id': 'pi_test_123456',
                'client_secret': 'test_secret',
                'status': 'requires_payment_method'
            })
        
        class error:
            class StripeError(Exception):
                pass
    
    monkeypatch.setattr("server.payment.payment_manager.stripe", MockStripe)
    return MockStripe


@pytest.fixture(scope="function")
def mock_openai(monkeypatch):
    """Mock OpenAI API responses."""
    class MockResponse:
        def __init__(self, status_code=200):
            self.status_code = status_code
            self.text = "Mock response"
            
        def json(self):
            return {
                "data": [{
                    "url": "https://example.com/mock-image.png"
                }]
            }
    
    def mock_post(*args, **kwargs):
        return MockResponse()
    
    def mock_get(*args, **kwargs):
        return MockResponse()
    
    monkeypatch.setattr("server.ai.model_handler.requests.post", mock_post)
    monkeypatch.setattr("server.ai.model_handler.requests.get", mock_get)
    
    return {"post": mock_post, "get": mock_get}


@pytest.fixture(scope="function")
def mock_torch(monkeypatch):
    """Mock PyTorch functionalities."""
    import numpy as np
    
    class MockTorchModule:
        def __init__(self):
            pass
            
        def to(self, device):
            return self
            
        def eval(self):
            return self
            
        def __call__(self, *args, **kwargs):
            # Return mock tensor output
            return type('MockTensor', (), {'detach': lambda: np.zeros((1, 10))})
    
    class MockTorch:
        @staticmethod
        def load(*args, **kwargs):
            return MockTorchModule()
            
        class device:
            @staticmethod
            def __call__(device_str):
                return device_str
                
        class no_grad:
            def __enter__(self):
                return None
                
            def __exit__(self, *args):
                pass
                
    monkeypatch.setattr("src.ai.model_handler.torch", MockTorch)
    return MockTorch