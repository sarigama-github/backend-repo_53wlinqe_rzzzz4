"""
Database Schemas for Eco Nanny

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercase of the class name (e.g., User -> "user").
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime


class User(BaseModel):
    """
    Users collection schema
    Collection name: "user"
    """
    google_id: str = Field(..., description="Google subject (sub) identifier")
    email: EmailStr = Field(..., description="Email address")
    name: str = Field(..., description="Full name")
    photo_url: Optional[str] = Field(None, description="Profile photo URL")
    role: str = Field(..., description="user role: 'nanny' | 'client' | 'admin'", pattern="^(nanny|client|admin)$")
    is_admin: bool = Field(False, description="Whether this user has admin access")


class Nannyprofile(BaseModel):
    """
    Nanny profiles
    Collection name: "nannyprofile"
    """
    user_id: str = Field(..., description="Owner user id (stringified ObjectId)")
    bio: Optional[str] = Field(None, description="About the nanny")
    years_experience: Optional[int] = Field(None, ge=0, le=60)
    hourly_rate: Optional[float] = Field(None, ge=0)
    skills: List[str] = Field(default_factory=list, description="List of skills/certifications")
    availability: Optional[str] = Field(None, description="Free text availability notes")
    location: Optional[str] = Field(None, description="City/Area")


class Clientprofile(BaseModel):
    """
    Client profiles
    Collection name: "clientprofile"
    """
    user_id: str = Field(..., description="Owner user id (stringified ObjectId)")
    child_name: Optional[str] = None
    child_age: Optional[int] = Field(None, ge=0, le=18)
    notes: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None


class Booking(BaseModel):
    """
    Bookings between clients and nannies
    Collection name: "booking"
    """
    client_id: str = Field(..., description="Client user id")
    nanny_id: str = Field(..., description="Nanny user id")
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)
    status: str = Field("pending", description="pending|accepted|declined|cancelled|completed",
                        pattern="^(pending|accepted|declined|cancelled|completed)$")
    notes: Optional[str] = None


class Message(BaseModel):
    """
    Chat messages per booking
    Collection name: "message"
    """
    booking_id: str = Field(...)
    sender_id: str = Field(...)
    text: str = Field(...)


class Review(BaseModel):
    """
    Reviews from clients to nannies
    Collection name: "review"
    """
    booking_id: str = Field(...)
    client_id: str = Field(...)
    nanny_id: str = Field(...)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
