"""
SQLAlchemy database models for the Th.ink AR Tattoo Visualization system.

This module defines the ORM models that represent the database schema
for the application, including relationships and constraints.
"""

from sqlalchemy import (
    Column, Integer, String, Float, ForeignKey, DateTime, Boolean, 
    Text, Enum, JSON, CheckConstraint, Index, UniqueConstraint, Table
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum
import uuid
from typing import List, Optional

Base = declarative_base()


# Association tables for many-to-many relationships
design_tags = Table(
    'design_tags',
    Base.metadata,
    Column('design_id', Integer, ForeignKey('tattoo_designs.id', ondelete='CASCADE'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True)
)

user_favorites = Table(
    'user_favorites',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('design_id', Integer, ForeignKey('tattoo_designs.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)


# Enum classes for type constraints
class SubscriptionTierEnum(enum.Enum):
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


class PaymentStatusEnum(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELED = "canceled"


class TattooStyleEnum(enum.Enum):
    TRADITIONAL = "traditional"
    REALISTIC = "realistic"
    BLACKWORK = "blackwork"
    WATERCOLOR = "watercolor"
    TRIBAL = "tribal"
    JAPANESE = "japanese" 
    NEW_SCHOOL = "new_school"
    MINIMALIST = "minimalist"
    GEOMETRIC = "geometric"
    DOTWORK = "dotwork"


class User(Base):
    """User model representing application users."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    subscription_tier = Column(
        Enum(SubscriptionTierEnum), 
        default=SubscriptionTierEnum.FREE, 
        nullable=False
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    profile_image_url = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    settings = Column(JSON, nullable=True)

    # Relationships
    designs = relationship("TattooDesign", back_populates="artist", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    favorite_designs = relationship(
        "TattooDesign", 
        secondary=user_favorites,
        back_populates="favorited_by"
    )

    # Index on is_active to quickly filter active users
    __table_args__ = (
        Index('idx_user_active', 'is_active'),
    )

    @validates('username')
    def validate_username(self, key, value):
        """Validate username length and format."""
        if not 3 <= len(value) <= 50:
            raise ValueError("Username must be between 3 and 50 characters")
        return value

    @validates('email')
    def validate_email(self, key, value):
        """Basic email validation."""
        if value is not None and '@' not in value:
            raise ValueError("Invalid email address")
        return value


class TattooDesign(Base):
    """Model for tattoo designs that can be previewed or purchased."""
    __tablename__ = "tattoo_designs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    image_url = Column(String(255), nullable=True)
    thumbnail_url = Column(String(255), nullable=True)
    style = Column(Enum(TattooStyleEnum), default=TattooStyleEnum.TRADITIONAL, nullable=False)
    
    # Foreign keys
    artist_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    is_ai_generated = Column(Boolean, default=False, nullable=False)
    rating = Column(Float, default=0.0, nullable=False)
    rating_count = Column(Integer, default=0, nullable=False)
    purchase_count = Column(Integer, default=0, nullable=False)
    view_count = Column(Integer, default=0, nullable=False)
    
    # Design properties
    width = Column(Integer, nullable=True)  # Width in pixels
    height = Column(Integer, nullable=True)  # Height in pixels
    colors = Column(JSON, nullable=True)  # Main colors used in the design
    
    # Relationships
    artist = relationship("User", back_populates="designs")
    tags = relationship("Tag", secondary=design_tags, back_populates="designs")
    transactions = relationship("Payment", back_populates="design")
    favorited_by = relationship(
        "User", 
        secondary=user_favorites,
        back_populates="favorite_designs"
    )
    reviews = relationship("DesignReview", back_populates="design", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint('price >= 0', name='check_positive_price'),
        Index('idx_tattoo_design_style', 'style'),
        Index('idx_tattoo_design_public', 'is_public'),
    )

    @validates('price')
    def validate_price(self, key, value):
        """Validate price is positive."""
        if value < 0:
            raise ValueError("Price must be non-negative")
        return value


class Tag(Base):
    """Tags for categorizing tattoo designs."""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    designs = relationship("TattooDesign", secondary=design_tags, back_populates="tags")

    __table_args__ = (
        Index('idx_tag_name', 'name'),
    )


class Payment(Base):
    """Model for payment transactions."""
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    status = Column(Enum(PaymentStatusEnum), nullable=False, default=PaymentStatusEnum.PENDING)
    external_payment_id = Column(String(255), nullable=True)
    payment_method = Column(String(50), nullable=True)
    
    # What was purchased
    design_id = Column(Integer, ForeignKey("tattoo_designs.id", ondelete="SET NULL"), nullable=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id", ondelete="SET NULL"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Additional metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="payments")
    design = relationship("TattooDesign", back_populates="transactions")
    subscription = relationship("Subscription", back_populates="payment")

    __table_args__ = (
        CheckConstraint('amount > 0', name='check_positive_amount'),
        Index('idx_payment_status', 'status'),
        Index('idx_payment_external_id', 'external_payment_id'),
    )

    @validates('amount')
    def validate_amount(self, key, value):
        """Validate amount is positive."""
        if value <= 0:
            raise ValueError("Payment amount must be positive")
        return value


class Subscription(Base):
    """Model for user subscriptions."""
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    subscription_tier = Column(Enum(SubscriptionTierEnum), nullable=False)
    status = Column(String(20), nullable=False, default="active")
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    auto_renew = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    canceled_at = Column(DateTime, nullable=True)
    
    # External references
    external_subscription_id = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    payment = relationship("Payment", back_populates="subscription", uselist=False)

    __table_args__ = (
        Index('idx_subscription_status', 'status'),
        Index('idx_subscription_end_date', 'end_date'),
    )


class UserSession(Base):
    """Model for tracking user sessions."""
    __tablename__ = "user_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 addresses can be up to 45 chars
    user_agent = Column(String(255), nullable=True)
    device_info = Column(JSON, nullable=True)
    
    # Session metrics
    designs_generated = Column(Integer, default=0)
    designs_viewed = Column(Integer, default=0)
    purchases_made = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index('idx_session_user', 'user_id'),
        Index('idx_session_start_time', 'start_time'),
    )


class DesignReview(Base):
    """Model for user reviews of tattoo designs."""
    __tablename__ = "design_reviews"

    id = Column(Integer, primary_key=True, index=True)
    design_id = Column(Integer, ForeignKey("tattoo_designs.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    rating = Column(Integer, nullable=False)
    review_text = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    design = relationship("TattooDesign", back_populates="reviews")
    
    __table_args__ = (
        CheckConstraint('rating >= 1 AND rating <= 5', name='check_valid_rating'),
        # Ensure a user can only review each design once
        UniqueConstraint('design_id', 'user_id', name='uq_user_design_review'),
    )

    @validates('rating')
    def validate_rating(self, key, value):
        """Validate rating is between 1 and 5."""
        if not 1 <= value <= 5:
            raise ValueError("Rating must be between 1 and 5")
        return value


class PrivacySettings(Base):
    """Model for user privacy settings."""
    __tablename__ = "privacy_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    allow_data_collection = Column(Boolean, default=True, nullable=False)
    allow_ai_training = Column(Boolean, default=True, nullable=False)
    allow_marketing = Column(Boolean, default=True, nullable=False)
    data_retention_days = Column(Integer, default=365, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('idx_privacy_user', 'user_id'),
    )


class NeRFAvatar(Base):
    """Model for NeRF Metahuman Avatar data."""
    __tablename__ = "nerf_avatars"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    model_url = Column(String(255), nullable=False)
    thumbnail_url = Column(String(255), nullable=True)
    
    # NeRF model metadata
    resolution = Column(Integer, default=512, nullable=False)
    is_animated = Column(Boolean, default=False, nullable=False)
    model_format = Column(String(50), default="nerf", nullable=False)
    model_size_mb = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Additional properties
    metadata = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_nerf_user', 'user_id'),
    )