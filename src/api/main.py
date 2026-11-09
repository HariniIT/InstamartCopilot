
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from src.api.schemas import ComplaintRequest, ComplaintResponse

from src.part_d.refund_agent_integration import process_customer_complaint


app = FastAPI(
    title="Instamart Copilot API",
    description="AI-powered Customer Support and Refund Assistant",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "Instamart Copilot",
        "message": "Welcome to Instamart Copilot API",
        "status": "Running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "Healthy",
        "service": "Instamart Copilot Backend",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/info")
def project_info():
    return {
        "project_name": "Instamart Copilot",
        "part_a": "Sales Analytics",
        "part_b": "Customer Segmentation and Churn Prediction",
        "part_c": "Customer Support Ticket Intelligence",
        "part_d": "AI Refund Policy Agent",
        "part_e": "FastAPI Backend",
        "status": "Development"
    }


@app.post(
    "/api/complaints",
    response_model=ComplaintResponse
)
def analyze_complaint(request: ComplaintRequest):

    try:
        result = process_customer_complaint(request.complaint)

        ticket = result["ticket"]
        policy = result["policy"]

        policy_id = policy.get("policy_id") if policy else None

        policy_title = (
            policy.get("policy_title") or policy.get("title")
            if policy else None
        )

        similarity = result["similarity"]

        return {
            "customer_id": request.customer_id,
            "complaint": request.complaint,
            "category": ticket["category"],
            "sentiment": ticket["sentiment"],
            "urgency": ticket["urgency"],
            "priority": ticket["priority"],
            "policy_id": policy_id,
            "policy_title": policy_title,
            "similarity": similarity,
            "ai_response": result["response"],
            "human_escalation": result["escalation_required"]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Complaint processing failed: {str(error)}"
        )
