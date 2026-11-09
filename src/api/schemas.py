
from pydantic import BaseModel, Field


class ComplaintRequest(BaseModel):
    customer_id: str = Field(
        default="CUST001",
        description="Customer ID"
    )

    complaint: str = Field(
        min_length=5,
        max_length=1000,
        description="Customer complaint message"
    )


class ComplaintResponse(BaseModel):
    customer_id: str
    complaint: str

    category: str
    sentiment: str
    urgency: str
    priority: str

    policy_id: str | None
    policy_title: str | None
    similarity: float | None

    ai_response: str
    human_escalation: bool
