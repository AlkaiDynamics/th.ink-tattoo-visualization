"""
Pydantic schema models for the Th.ink AR Tattoo Visualization system.

This module defines the data validation schemas used throughout the application
for API request/response models, ensuring type safety and data integrity.
"""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator, constr, conint, confloat
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from uuid import UUID


# Shared configuration
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Enums for validation
class SubscriptionTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELED = "canceled"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"


class TattooStyle(str, Enum):
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


# User Schemas
class UserBase(BaseModel):
    """Base model for user data with common fields."""
    username: constr(min_length=3, max_length=50) = Field(
        ..., description="Username for login, 3-50 characters"
    )
    email: Optional[EmailStr] = Field(
        None, description="User's email address, optional but recommended"
    )
    full_name: Optional[constr(max_length=100)] = Field(
        None, description="User's full name"
    )


class UserCreate(UserBase):
    """Schema for user creation including password."""
    password: constr(min_length=8) = Field(
        ..., description="User password, minimum 8 characters"
    )

    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

    class Config:
        schema_extra = {
            "example": {
                "username": "tattoo_enthusiast",
                "email": "user@example.com",
                "full_name": "Jane Doe",
                "password": "SecurePass123"
            }
        }


class UserRead(UserBase):
    """Schema for user information retrieval, excluding sensitive data."""
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    subscription_tier: Optional[SubscriptionTier] = SubscriptionTier.FREE

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "username": "tattoo_enthusiast",
                "email": "user@example.com",
                "full_name": "Jane Doe",
                "created_at": "2023-01-15T12:00:00",
                "updated_at": "2023-01-15T12:00:00",
                "subscription_tier": "free"
            }
        }


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    username: Optional[constr(min_length=3, max_length=50)] = None
    email: Optional[EmailStr] = None
    full_name: Optional[constr(max_length=100)] = None
    password: Optional[constr(min_length=8)] = None

    @validator('password')
    def password_strength(cls, v):
        """Validate password strength if provided."""
        if v is None:
            return v
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

    class Config:
        schema_extra = {
            "example": {
                "email": "newemail@example.com",
                "full_name": "Jane Smith",
            }
        }


# Authentication Schemas
class Token(BaseModel):
    """Schema for authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    user_id: int

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
                "user_id": 1
            }
        }


class TokenPayload(BaseModel):
    """Schema for token payload data."""
    sub: Union[int, str]
    exp: int


# Tattoo Design Schemas
class TattooDesignBase(BaseModel):
    """Base model for tattoo design data."""
    name: constr(min_length=1, max_length=100) = Field(
        ..., description="Name of the tattoo design"
    )
    description: Optional[str] = Field(
        None, description="Description of the tattoo design"
    )
    price: confloat(ge=0.0) = Field(
        ..., description="Price of the tattoo design in USD"
    )
    style: TattooStyle = Field(
        TattooStyle.TRADITIONAL, description="Style of the tattoo"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing the design"
    )


class TattooDesignCreate(TattooDesignBase):
    """Schema for tattoo design creation."""
    image_url: Optional[str] = Field(
        None, description="URL to the tattoo design image"
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "Phoenix Rising",
                "description": "A traditional phoenix design symbolizing rebirth",
                "price": 49.99,
                "style": "traditional",
                "tags": ["phoenix", "bird", "fire", "rebirth"],
                "image_url": "https://example.com/phoenix.png"
            }
        }


class TattooDesignUpdate(BaseModel):
    """Schema for updating tattoo design information."""
    name: Optional[constr(min_length=1, max_length=100)] = None
    description: Optional[str] = None
    price: Optional[confloat(ge=0.0)] = None
    style: Optional[TattooStyle] = None
    tags: Optional[List[str]] = None
    image_url: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "price": 59.99,
                "description": "An updated description for the phoenix design"
            }
        }


class TattooDesignRead(TattooDesignBase):
    """Schema for tattoo design retrieval."""
    id: int
    artist_id: int
    image_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    rating: Optional[float] = None
    purchase_count: Optional[int] = 0

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Phoenix Rising",
                "description": "A traditional phoenix design symbolizing rebirth",
                "price": 49.99,
                "style": "traditional",
                "tags": ["phoenix", "bird", "fire", "rebirth"],
                "artist_id": 1,
                "image_url": "https://example.com/phoenix.png",
                "created_at": "2023-01-15T12:00:00",
                "updated_at": "2023-01-15T12:00:00",
                "rating": 4.8,
                "purchase_count": 42
            }
        }


# Payment Schemas
class PaymentBase(BaseModel):
    """Base model for payment data."""
    amount: confloat(gt=0) = Field(
        ..., description="Payment amount"
    )
    currency: Currency = Field(
        Currency.USD, description="Payment currency"
    )


class PaymentCreate(PaymentBase):
    """Schema for payment creation."""
    design_id: Optional[int] = Field(
        None, description="ID of the tattoo design being purchased, if applicable"
    )
    subscription_tier: Optional[SubscriptionTier] = Field(
        None, description="Subscription tier being purchased, if applicable"
    )

    @validator('subscription_tier', 'design_id')
    def validate_payment_target(cls, v, values, **kwargs):
        """Validate that either design_id or subscription_tier is provided."""
        field = kwargs.get('field')
        if field.name == 'subscription_tier':
            design_id = values.get('design_id')
            if v is None and design_id is None:
                raise ValueError('Either design_id or subscription_tier must be provided')
        return v

    class Config:
        schema_extra = {
            "example": {
                "amount": 49.99,
                "currency": "USD",
                "design_id": 1
            }
        }


class PaymentRead(PaymentBase):
    """Schema for payment retrieval."""
    id: int
    user_id: int
    status: PaymentStatus
    external_payment_id: Optional[str] = None
    design_id: Optional[int] = None
    subscription_tier: Optional[SubscriptionTier] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "amount": 49.99,
                "currency": "USD",
                "user_id": 1,
                "status": "completed",
                "external_payment_id": "pi_1J2DfgHJd8z9KY2T4QjgL2Br",
                "design_id": 1,
                "subscription_tier": None,
                "created_at": "2023-01-15T12:00:00",
                "updated_at": "2023-01-15T12:05:00"
            }
        }


# Subscription Schemas
class SubscriptionBase(BaseModel):
    """Base model for subscription data."""
    subscription_tier: SubscriptionTier = Field(
        ..., description="Subscription tier level"
    )


class SubscriptionCreate(SubscriptionBase):
    """Schema for subscription creation."""
    user_id: int = Field(
        ..., description="ID of the user for the subscription"
    )
    start_date: datetime = Field(
        ..., description="Subscription start date"
    )
    end_date: datetime = Field(
        ..., description="Subscription end date"
    )

    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        """Validate that end_date is after start_date."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "subscription_tier": "premium",
                "start_date": "2023-01-15T00:00:00",
                "end_date": "2024-01-15T00:00:00"
            }
        }


class SubscriptionUpdate(BaseModel):
    """Schema for updating subscription information."""
    subscription_tier: Optional[SubscriptionTier] = None
    status: Optional[str] = None
    end_date: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "subscription_tier": "pro",
                "end_date": "2024-01-15T00:00:00"
            }
        }


class SubscriptionRead(SubscriptionBase):
    """Schema for subscription retrieval."""
    id: int
    user_id: int
    start_date: datetime
    end_date: datetime
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "subscription_tier": "premium",
                "start_date": "2023-01-15T00:00:00",
                "end_date": "2024-01-15T00:00:00",
                "status": "active",
                "created_at": "2023-01-15T12:00:00",
                "updated_at": "2023-01-15T12:00:00"
            }
        }


# AI-generated tattoo request schema
class TattooGeneratorRequest(BaseModel):
    """Schema for tattoo generation requests."""
    description: constr(min_length=3, max_length=500) = Field(
        ..., description="Text description of the desired tattoo"
    )
    style: TattooStyle = Field(
        TattooStyle.TRADITIONAL, description="Preferred tattoo style"
    )
    size: Literal["small", "medium", "large"] = Field(
        "medium", description="Desired tattoo size"
    )
    color: bool = Field(
        True, description="Whether the tattoo should be in color"
    )

    class Config:
        schema_extra = {
            "example": {
                "description": "A phoenix rising from ashes with flames in red and orange",
                "style": "traditional",
                "size": "medium",
                "color": True
            }
        }


class TattooGeneratorResponse(BaseModel):
    """Schema for tattoo generation responses."""
    image_url: str = Field(
        ..., description="URL to the generated tattoo image"
    )
    design_id: Optional[int] = Field(
        None, description="ID of the saved design, if available"
    )
    generation_time: float = Field(
        ..., description="Time taken to generate the design in seconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://storage.think-ar.com/generated/phoenix-12345.png",
                "design_id": None,
                "generation_time": 3.45
            }
        }


# Collection schemas for list responses
class TattooDesignList(BaseModel):
    """Schema for list of tattoo designs."""
    designs: List[TattooDesignRead]
    total: int
    page: int
    per_page: int
    pages: int

    class Config:
        schema_extra = {
            "example": {
                "designs": [
                    {
                        "id": 1,
                        "name": "Phoenix Rising",
                        "price": 49.99,
                        "artist_id": 1
                    }
                ],
                "total": 42,
                "page": 1,
                "per_page": 20,
                "pages": 3
            }
        }


class PaymentList(BaseModel):
    """Schema for list of payments."""
    payments: List[PaymentRead]
    total: int
    page: int
    per_page: int

    class Config:
        schema_extra = {
            "example": {
                "payments": [
                    {
                        "id": 1,
                        "amount": 49.99,
                        "currency": "USD",
                        "status": "completed"
                    }
                ],
                "total": 5,
                "page": 1,
                "per_page": 20
            }
        }


# Error response schema
class HTTPError(BaseModel):
    """Schema for HTTP error responses."""
    detail: str
    status_code: int
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "detail": "Not found",
                "status_code": 404,
                "error_code": "RESOURCE_NOT_FOUND",
                "timestamp": "2023-01-15T12:00:00"
            }
        }