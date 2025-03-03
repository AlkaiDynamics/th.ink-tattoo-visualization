from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List

# User Schemas
class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id: int

    class Config:
        orm_mode = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

# Tattoo Design Schemas
class TattooDesignBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

class TattooDesignCreate(TattooDesignBase):
    pass

class TattooDesignUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None

class TattooDesignRead(TattooDesignBase):
    id: int
    artist_id: int

    class Config:
        orm_mode = True

# Payment Schemas
class PaymentBase(BaseModel):
    amount: float
    currency: str

class PaymentRead(PaymentBase):
    id: int
    user_id: int
    status: str
    external_payment_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class PaymentCreate(PaymentBase):
    pass

# Additional Schemas if needed
class TattooDesignList(BaseModel):
    designs: List[TattooDesignRead]

class PaymentList(BaseModel):
    payments: List[PaymentRead]