"""Decidr Coherence Engine — Schemas (v4)"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

BUCKET_HIERARCHY = {
    "Marketing": {
        "Paid Acquisition": ["Google Ads - Search", "Google Ads - Display", "Social Media Ads"],
        "Content & SEO": ["Blog Content Production", "SEO Optimization", "Video Content"],
        "Partnerships": ["Partner Acquisition", "Co-Marketing Campaigns", "Channel Partnerships"],
        "Events & Community": ["Virtual Events", "In-Person Events", "Community Management"],
    },
    "Product": {
        "Core Platform": ["Backend Services", "Frontend Development", "APIs & Integrations"],
        "New Features": ["Feature Development", "A/B Testing", "Feature Launch"],
        "Technical Debt": ["Code Refactoring", "Infrastructure Upgrades"],
        "Research & Innovation": ["Proof of Concepts", "Innovation Labs"],
    },
    "Operations": {
        "Customer Success": ["Onboarding", "Support Tickets", "Account Management"],
        "Infrastructure & DevOps": ["Cloud Infrastructure", "CI/CD Pipeline"],
        "Data & Analytics": ["Data Engineering", "Analytics & Reporting"],
    },
    "G&A": {
        "Finance & Legal": ["Financial Planning", "Legal & Compliance"],
        "HR & Recruiting": ["Talent Acquisition", "Employee Development"],
        "General Admin": ["Office Operations", "Vendor Management"],
    },
}
METRIC_UNITS = ["score","percentage","ratio","count","dollars","visitors","leads","attendees","ms","seconds","hours","days","projects"]
SCENARIOS = ["optimal","underfunded","overfunded","dynamic"]
SHOCK_TYPES = ["none", "budget_shock", "market_shock"]
DIMS = ["coherence","attainability","relevance","integrity"]
WEIGHTS = {"coherence":0.25,"attainability":0.25,"relevance":0.25,"integrity":0.25}
def get_l2_for_l1(l1): return list(BUCKET_HIERARCHY.get(l1,{}).keys())
def get_l3_for_l2(l1,l2): return BUCKET_HIERARCHY.get(l1,{}).get(l2,[])

class GoalInput(BaseModel):
    goal_title: str = ""
    scope: str = "goal"
    bucket_l1: str = ""
    bucket_l2: str = ""
    bucket_l3: Optional[str] = None
    metric_name: str = ""
    metric_unit: str = "score"
    target_value: Optional[float] = None
    initial_value: Optional[float] = None
    periods: int = 24
    scenario_story: str = "optimal"
    shock_type: str = "none"

class System3Payload(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    goal: GoalInput = Field(default_factory=GoalInput)
    scope: str = "goal"
    raw_nl_input: str = ""
    metadata: dict = Field(default_factory=lambda: {"source":"system_1","version":"0.4.0"})
    def to_json(self): return self.model_dump_json(indent=2)