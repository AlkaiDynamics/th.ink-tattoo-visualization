from fastapi import FastAPI, HTTPException
from server.database import models, engine

models.Base.metadata.create_all(bind=engine)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Th.ink AI-powered Tattoo Visualization System",
    description="Backend services for subscription/payment logic, user management, and AI microservices.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Th.ink Backend API"}

# Include routers for different modules
from server.auth.auth_manager import auth_router
from server.payment.payment_manager import payment_router
from server.marketplace.marketplace_manager import marketplace_router

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(payment_router, prefix="/payments", tags=["Payments"])
app.include_router(marketplace_router, prefix="/marketplace", tags=["Marketplace"])

# Add more routes and functionalities as needed