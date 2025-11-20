import os
from datetime import datetime, timezone
from typing import List, Optional

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from database import db, create_document, get_documents

app = FastAPI(title="Eco Nanny API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Utils & Auth
# -------------------------
ADMIN_EMAILS = set([
    e.strip().lower() for e in (os.getenv("ADMIN_EMAILS", "").split(",") if os.getenv("ADMIN_EMAILS") else [])
])


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """Simple auth using Authorization: Bearer <user_id>
    After Google sign-in, we return user_id. Client sends it as bearer on each request.
    """
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization scheme")
    user = db["user"].find_one({"_id": __import__("bson").ObjectId(token)}) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user token")
    # convert _id to str
    user["id"] = str(user.pop("_id"))
    return user


# -------------------------
# Schemas (request models)
# -------------------------
class GoogleAuthPayload(BaseModel):
    id_token: str = Field(..., description="Google ID token from GIS")
    role: str = Field(..., pattern="^(nanny|client)$")


class NannyProfileIn(BaseModel):
    bio: Optional[str] = None
    years_experience: Optional[int] = Field(None, ge=0, le=60)
    hourly_rate: Optional[float] = Field(None, ge=0)
    skills: List[str] = Field(default_factory=list)
    availability: Optional[str] = None
    location: Optional[str] = None


class ClientProfileIn(BaseModel):
    child_name: Optional[str] = None
    child_age: Optional[int] = Field(None, ge=0, le=18)
    notes: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None


class BookingIn(BaseModel):
    nanny_id: str
    start_time: datetime
    end_time: datetime
    notes: Optional[str] = None


class BookingStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(pending|accepted|declined|cancelled|completed)$")


class MessageIn(BaseModel):
    text: str


class ReviewIn(BaseModel):
    booking_id: str
    nanny_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


# -------------------------
# Basic routes
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Eco Nanny API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
        else:
            response["database"] = "❌ Not Available"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# -------------------------
# Google Auth
# -------------------------
@app.post("/auth/google")
def google_auth(payload: GoogleAuthPayload):
    """Verify Google ID token, upsert user, assign role, set admin flag if email allowed.
    Returns { access_token: user_id, user: {...} }
    """
    # Verify token with Google
    verify_url = "https://oauth2.googleapis.com/tokeninfo"
    r = requests.get(verify_url, params={"id_token": payload.id_token}, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    data = r.json()
    sub = data.get("sub")
    email = data.get("email")
    name = data.get("name") or data.get("given_name") or "User"
    picture = data.get("picture")
    if not sub or not email:
        raise HTTPException(status_code=401, detail="Google token missing subject/email")

    is_admin = email.lower() in ADMIN_EMAILS
    # Upsert user
    existing = db["user"].find_one({"google_id": sub})
    now = datetime.now(timezone.utc)
    if existing:
        db["user"].update_one(
            {"_id": existing["_id"]},
            {"$set": {"email": email, "name": name, "photo_url": picture, "role": payload.role, "is_admin": is_admin, "updated_at": now}},
        )
        user_id = str(existing["_id"])
    else:
        user_doc = {
            "google_id": sub,
            "email": email,
            "name": name,
            "photo_url": picture,
            "role": payload.role,
            "is_admin": is_admin,
            "created_at": now,
            "updated_at": now,
        }
        from bson import ObjectId
        result = db["user"].insert_one(user_doc)
        user_id = str(result.inserted_id)
    user = db["user"].find_one({"_id": __import__("bson").ObjectId(user_id)})
    user["id"] = str(user.pop("_id"))
    return {"access_token": user_id, "user": user}


@app.get("/me")
def me(user: dict = Depends(get_current_user)):
    return user


# -------------------------
# Profiles
# -------------------------
@app.get("/profile/nanny/me")
def get_nanny_profile(user: dict = Depends(get_current_user)):
    if user.get("role") != "nanny":
        raise HTTPException(403, detail="Only nannies can access this")
    prof = db["nannyprofile"].find_one({"user_id": user["id"]})
    if not prof:
        return None
    prof["id"] = str(prof.pop("_id"))
    return prof


@app.post("/profile/nanny")
def upsert_nanny_profile(payload: NannyProfileIn, user: dict = Depends(get_current_user)):
    if user.get("role") != "nanny":
        raise HTTPException(403, detail="Only nannies can edit this")
    now = datetime.now(timezone.utc)
    existing = db["nannyprofile"].find_one({"user_id": user["id"]})
    doc = {**payload.model_dump(), "user_id": user["id"], "updated_at": now}
    if existing:
        db["nannyprofile"].update_one({"_id": existing["_id"]}, {"$set": doc})
        _id = existing["_id"]
    else:
        doc["created_at"] = now
        _id = db["nannyprofile"].insert_one(doc).inserted_id
    prof = db["nannyprofile"].find_one({"_id": _id})
    prof["id"] = str(prof.pop("_id"))
    return prof


@app.get("/profile/client/me")
def get_client_profile(user: dict = Depends(get_current_user)):
    if user.get("role") != "client":
        raise HTTPException(403, detail="Only clients can access this")
    prof = db["clientprofile"].find_one({"user_id": user["id"]})
    if not prof:
        return None
    prof["id"] = str(prof.pop("_id"))
    return prof


@app.post("/profile/client")
def upsert_client_profile(payload: ClientProfileIn, user: dict = Depends(get_current_user)):
    if user.get("role") != "client":
        raise HTTPException(403, detail="Only clients can edit this")
    now = datetime.now(timezone.utc)
    existing = db["clientprofile"].find_one({"user_id": user["id"]})
    doc = {**payload.model_dump(), "user_id": user["id"], "updated_at": now}
    if existing:
        db["clientprofile"].update_one({"_id": existing["_id"]}, {"$set": doc})
        _id = existing["_id"]
    else:
        doc["created_at"] = now
        _id = db["clientprofile"].insert_one(doc).inserted_id
    prof = db["clientprofile"].find_one({"_id": _id})
    prof["id"] = str(prof.pop("_id"))
    return prof


# -------------------------
# Bookings
# -------------------------
@app.post("/bookings")
def create_booking(payload: BookingIn, user: dict = Depends(get_current_user)):
    if user.get("role") != "client":
        raise HTTPException(403, detail="Only clients can create bookings")
    now = datetime.now(timezone.utc)
    doc = {
        "client_id": user["id"],
        "nanny_id": payload.nanny_id,
        "start_time": payload.start_time,
        "end_time": payload.end_time,
        "status": "pending",
        "notes": payload.notes,
        "created_at": now,
        "updated_at": now,
    }
    _id = db["booking"].insert_one(doc).inserted_id
    b = db["booking"].find_one({"_id": _id})
    b["id"] = str(b.pop("_id"))
    return b


@app.get("/bookings")
def list_bookings(user: dict = Depends(get_current_user)):
    role = user.get("role")
    query = {"client_id": user["id"]} if role == "client" else {"nanny_id": user["id"]}
    items = []
    for b in db["booking"].find(query).sort("created_at", -1):
        b["id"] = str(b.pop("_id"))
        items.append(b)
    return items


@app.patch("/bookings/{booking_id}")
def update_booking_status(booking_id: str, payload: BookingStatusUpdate, user: dict = Depends(get_current_user)):
    from bson import ObjectId
    b = db["booking"].find_one({"_id": ObjectId(booking_id)})
    if not b:
        raise HTTPException(404, detail="Booking not found")
    # permissions
    if user["role"] == "nanny" and user["id"] != b["nanny_id"]:
        raise HTTPException(403, detail="Not your booking")
    if user["role"] == "client" and user["id"] != b["client_id"]:
        raise HTTPException(403, detail="Not your booking")
    db["booking"].update_one({"_id": b["_id"]}, {"$set": {"status": payload.status, "updated_at": datetime.now(timezone.utc)}})
    b = db["booking"].find_one({"_id": b["_id"]})
    b["id"] = str(b.pop("_id"))
    return b


# -------------------------
# Messages
# -------------------------
@app.get("/messages/{booking_id}")
def list_messages(booking_id: str, user: dict = Depends(get_current_user)):
    from bson import ObjectId
    b = db["booking"].find_one({"_id": ObjectId(booking_id)})
    if not b:
        raise HTTPException(404, detail="Booking not found")
    if user["id"] not in (b["client_id"], b["nanny_id"]):
        raise HTTPException(403, detail="No access")
    msgs = []
    for m in db["message"].find({"booking_id": booking_id}).sort("created_at", 1):
        m["id"] = str(m.pop("_id"))
        msgs.append(m)
    return msgs


@app.post("/messages/{booking_id}")
def send_message(booking_id: str, payload: MessageIn, user: dict = Depends(get_current_user)):
    from bson import ObjectId
    b = db["booking"].find_one({"_id": ObjectId(booking_id)})
    if not b:
        raise HTTPException(404, detail="Booking not found")
    if user["id"] not in (b["client_id"], b["nanny_id"]):
        raise HTTPException(403, detail="No access")
    now = datetime.now(timezone.utc)
    doc = {"booking_id": booking_id, "sender_id": user["id"], "text": payload.text, "created_at": now}
    _id = db["message"].insert_one(doc).inserted_id
    msg = db["message"].find_one({"_id": _id})
    msg["id"] = str(msg.pop("_id"))
    # Simple broadcast to WebSocket connections if any
    ChatManager.broadcast(booking_id, {"type": "message", "data": msg})
    return msg


# -------------------------
# Reviews
# -------------------------
@app.post("/reviews")
def create_review(payload: ReviewIn, user: dict = Depends(get_current_user)):
    # only clients can review and only for completed bookings they participated in
    if user["role"] != "client":
        raise HTTPException(403, detail="Only clients can review")
    from bson import ObjectId
    b = db["booking"].find_one({"_id": ObjectId(payload.booking_id)})
    if not b or b.get("status") != "completed" or b.get("client_id") != user["id"] or b.get("nanny_id") != payload.nanny_id:
        raise HTTPException(400, detail="Invalid booking for review")
    now = datetime.now(timezone.utc)
    doc = {
        "booking_id": payload.booking_id,
        "client_id": user["id"],
        "nanny_id": payload.nanny_id,
        "rating": payload.rating,
        "comment": payload.comment,
        "created_at": now,
    }
    _id = db["review"].insert_one(doc).inserted_id
    rv = db["review"].find_one({"_id": _id})
    rv["id"] = str(rv.pop("_id"))
    return rv


@app.get("/reviews/nanny/{nanny_id}")
def list_reviews_for_nanny(nanny_id: str):
    rvs = []
    for r in db["review"].find({"nanny_id": nanny_id}).sort("created_at", -1):
        r["id"] = str(r.pop("_id"))
        rvs.append(r)
    return rvs


# -------------------------
# Admin
# -------------------------
@app.get("/admin/overview")
def admin_overview(user: dict = Depends(get_current_user)):
    if not user.get("is_admin"):
        raise HTTPException(403, detail="Admins only")
    return {
        "users": db["user"].count_documents({}),
        "nannies": db["user"].count_documents({"role": "nanny"}),
        "clients": db["user"].count_documents({"role": "client"}),
        "bookings": db["booking"].count_documents({}),
        "reviews": db["review"].count_documents({}),
    }


# -------------------------
# WebSocket Chat
# -------------------------
class ChatManager:
    connections = {}  # booking_id -> set of websockets

    @classmethod
    def register(cls, booking_id: str, ws: WebSocket):
        cls.connections.setdefault(booking_id, set()).add(ws)

    @classmethod
    def unregister(cls, booking_id: str, ws: WebSocket):
        if booking_id in cls.connections and ws in cls.connections[booking_id]:
            cls.connections[booking_id].remove(ws)
            if not cls.connections[booking_id]:
                del cls.connections[booking_id]

    @classmethod
    def broadcast(cls, booking_id: str, message: dict):
        # Send to all connections for this booking; ignore send errors
        if booking_id not in cls.connections:
            return
        to_remove = []
        for ws in list(cls.connections[booking_id]):
            try:
                import anyio
                anyio.from_thread.run(ws.send_json, message)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            cls.unregister(booking_id, ws)


@app.websocket("/ws/chat/{booking_id}")
async def websocket_endpoint(websocket: WebSocket, booking_id: str):
    await websocket.accept()
    ChatManager.register(booking_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back; in normal flow, messages are posted via REST and broadcasted
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        ChatManager.unregister(booking_id, websocket)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
