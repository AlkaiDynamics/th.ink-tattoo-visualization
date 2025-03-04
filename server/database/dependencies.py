"""
Database dependencies and connection management for the Th.ink AR application.

This module provides the database connection setup, session management,
and dependency injection for database access throughout the application.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import os
import logging
from typing import Generator, Optional
from functools import lru_cache
from contextlib import contextmanager

from ..config.model_config import get_config

# Configure logger
logger = logging.getLogger("think.database")

# Get configuration
config = get_config()

# Use the DATABASE_URL from environment variables with a fallback
DATABASE_URL = os.getenv('DATABASE_URL', config.database.url)

# Engine configuration parameters
ENGINE_PARAMS = {
    'pool_size': config.database.pool_size,
    'max_overflow': config.database.max_overflow,
    'pool_timeout': 30,  # seconds
    'pool_recycle': 1800,  # 30 minutes
    'pool_pre_ping': True,  # Verify connections before using them
    'echo': config.database.echo,  # Set to True for SQL query logging
}

# Add SQLite-specific parameters if using SQLite
if DATABASE_URL.startswith("sqlite"):
    ENGINE_PARAMS['connect_args'] = {"check_same_thread": False}
    # For SQLite, we don't need pooling
    engine = create_engine(DATABASE_URL, **ENGINE_PARAMS)
else:
    # For other databases, use connection pooling
    engine = create_engine(
        DATABASE_URL, 
        poolclass=QueuePool,
        **ENGINE_PARAMS
    )

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models - imported from models.py
from .models import Base

# Add connection pool events for monitoring
@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    logger.debug("Database connection established")

@event.listens_for(engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    logger.debug("Database connection retrieved from pool")

@event.listens_for(engine, "checkin")
def checkin(dbapi_connection, connection_record):
    logger.debug("Database connection returned to pool")

@lru_cache()
def get_engine():
    """
    Get SQLAlchemy engine instance (cached).
    
    Returns:
        The SQLAlchemy engine singleton
    """
    return engine

def get_db() -> Generator[Session, None, None]:
    """
    Get a database session to use as a dependency in FastAPI routes.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        logger.debug("Creating new database session")
        yield db
    finally:
        logger.debug("Closing database session")
        db.close()

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Use this for operations outside of request handlers.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_database_tables() -> None:
    """
    Create all database tables if they don't exist.
    
    This should be called during application startup.
    """
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def drop_database_tables() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data! Only use in testing or with extreme caution.
    """
    if os.getenv("ENVIRONMENT", "production").lower() == "production":
        raise RuntimeError("Cannot drop tables in production environment")
    
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise

def check_database_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection is working, False otherwise
    """
    try:
        # Try to connect and execute a simple query
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

def get_connection_info() -> dict:
    """
    Get information about the database connection.
    
    Returns:
        dict: Connection information
    """
    return {
        "url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,  # Hide credentials
        "pool_size": ENGINE_PARAMS.get("pool_size"),
        "max_overflow": ENGINE_PARAMS.get("max_overflow"),
        "pool_timeout": ENGINE_PARAMS.get("pool_timeout"),
        "pool_recycle": ENGINE_PARAMS.get("pool_recycle"),
        "pool_pre_ping": ENGINE_PARAMS.get("pool_pre_ping"),
        "dialect": engine.dialect.name,
        "driver": engine.dialect.driver,
        "echo": ENGINE_PARAMS.get("echo")
    }