from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import classify_ticket, TicketClassification

app = FastAPI(title="HR Ticket Classifier Agent", version="1.0")


class TicketRequest(BaseModel):
    subject: str
    description: str


@app.get("/")
def health():
    return {"status": "ok", "service": "HR Ticket Classifier"}


@app.post("/classify", response_model=TicketClassification)
def classify(ticket: TicketRequest):
    try:
        return classify_ticket(ticket.subject, ticket.description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
