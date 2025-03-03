import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.app import app
from server.database.models import Base
from server.database.dependencies import get_db

# Create a new SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

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
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "full_name": "Test User",
        "password": "testpassword"
    }

def test_register_user_success(test_user):
    response = client.post("/auth/register", json=test_user)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == test_user["username"]
    assert data["email"] == test_user["email"]
    assert "id" in data

def test_register_user_existing_username(test_user):
    response = client.post("/auth/register", json=test_user)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Username already registered"

def test_register_user_existing_email():
    new_user = {
        "username": "newuser",
        "email": "testuser@example.com",
        "full_name": "New User",
        "password": "newpassword"
    }
    response = client.post("/auth/register", json=new_user)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Email already registered"

def test_login_user_success(test_user):
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_user_wrong_password(test_user):
    login_data = {
        "username": test_user["username"],
        "password": "wrongpassword"
    }
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Incorrect username or password"

def test_login_user_nonexistent_user():
    login_data = {
        "username": "nonexistent",
        "password": "password"
    }
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Incorrect username or password"