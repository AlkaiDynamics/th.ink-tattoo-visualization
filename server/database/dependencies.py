from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from .models import Base

# Database URL should be set in environment variables for security
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./think.db')

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()