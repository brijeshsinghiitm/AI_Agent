from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import classify_ticket, TicketClassification

BASE_DIR = Path(__file__).parent

app = FastAPI(title="HR Ticket Classifier Agent", version="1.1")


class TicketRequest(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=3, max_length=3000)


@app.get("/", include_in_schema=False)
def home():
    """Serves the web UI."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "HR Ticket Classifier"}


@app.post("/classify", response_model=TicketClassification)
def classify(ticket: TicketRequest):
    try:
        return classify_ticket(ticket.subject.strip(), ticket.description.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
