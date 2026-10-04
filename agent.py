from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # loads Gemini + LangSmith settings from .env


class TicketCategory(str, Enum):
    RECRUITMENT = "Recruitment"
    LND = "Learning and Development"
    HRBP = "HR Business Partner"
    HR_OPS = "HR Operations"
    HR_POLICY = "HR Policy"


class TicketClassification(BaseModel):
    category: TicketCategory = Field(description="The single best-fit HR category")
    confidence: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    reasoning: str = Field(description="One-sentence justification")


SYSTEM_PROMPT = """You are an HR ticket triage agent. Classify each ticket into exactly one category:

- Recruitment: job openings, hiring requests, interview scheduling, referrals, offers, candidate queries, onboarding of new hires pre-joining.
- Learning and Development: training requests, certifications, upskilling, courses, mentoring, skill gap, learning platforms.
- HR Business Partner: manager/employee relations, performance issues, conflicts, team restructuring, retention, career growth discussions, org design.
- HR Operations: payroll, payslips, leave balance, attendance, benefits administration, employee letters, ID cards, exit formalities, HRIS data changes.
- HR Policy: questions about policy rules or entitlements (leave policy, WFH policy, code of conduct, travel policy), policy clarifications or change requests.

If a ticket touches several areas, choose the PRIMARY intent. Be concise."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Ticket subject: {subject}\n\nTicket description: {description}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Structured output forces Gemini to return valid JSON matching our schema
classifier_chain = prompt | llm.with_structured_output(TicketClassification)


def classify_ticket(subject: str, description: str) -> TicketClassification:
    return classifier_chain.invoke(
        {"subject": subject, "description": description},
        config={"run_name": "hr_ticket_classification", "tags": ["hr", "triage"]},
    )
