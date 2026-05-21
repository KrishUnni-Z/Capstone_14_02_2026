from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# -----------------------------
# Incoming request: System 1 -> System 3
# -----------------------------

class ParsedRequest(BaseModel):
    """
    Parsed JSON coming from System 1.
    """
    model_config = ConfigDict(extra="ignore")

    request_type: str = Field(default="score_goals")
    period_id: int = 12
    goal_ids: Optional[List[int]] = None
    top_k: Optional[int] = None
    user_query: Optional[str] = None


class System1ToSystem3Payload(BaseModel):
    """
    Full payload sent from System 1 to System 3.
    """
    model_config = ConfigDict(extra="ignore")

    raw_user_input: str
    parsed_request: ParsedRequest


# -----------------------------
# Goal record: System 3 internal structured row
# -----------------------------

class GoalRecord(BaseModel):
    """
    One structured goal-level record from analytical_flat.
    """
    model_config = ConfigDict(extra="ignore")

    goal_id: int
    period_id: int

    bucket_id: Optional[int] = None
    parent_bucket_id: Optional[int] = None
    projection_id: Optional[int] = None

    goal_name: Optional[str] = None
    bucket_name: Optional[str] = None
    parent_bucket_name: Optional[str] = None
    metric_name: Optional[str] = None
    metric_unit: Optional[str] = None
    scenario_story: Optional[str] = None

    start_period: Optional[int] = None
    end_period: Optional[int] = None

    target_value_final_period: Optional[float] = None
    initial_value: Optional[float] = None
    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    variance_from_target: Optional[float] = None

    allocated_amount: Optional[float] = None
    allocated_time_hours: Optional[float] = None
    allocation_percentage_of_total: Optional[float] = None
    allocation_percentage_of_parent: Optional[float] = None

    minimum_viable_allocation: Optional[float] = None
    optimal_allocation_min: Optional[float] = None
    optimal_allocation_max: Optional[float] = None
    red_low_max: Optional[float] = None
    orange_low_max: Optional[float] = None
    green_min: Optional[float] = None
    green_max: Optional[float] = None
    orange_high_min: Optional[float] = None
    red_high_min: Optional[float] = None

    trailing_3_period_slope: Optional[float] = None
    trailing_6_period_slope: Optional[float] = None
    volatility_measure: Optional[float] = None

    delivered_output_quantity: Optional[float] = None
    delivered_output_quality_score: Optional[float] = None
    output_cost_per_unit: Optional[float] = None
    total_cost: Optional[float] = None

    range_position_score: Optional[float] = None
    status_band: Optional[str] = None
    underfunded_flag: Optional[bool] = None
    overfunded_flag: Optional[bool] = None
    allocation_efficiency_ratio: Optional[float] = None
    probability_of_hitting_target: Optional[float] = None
    time_to_green_estimate: Optional[float] = None


# -----------------------------
# Payload sent: System 3 -> System 2
# -----------------------------

class System2InputPayload(BaseModel):
    """
    Validated payload sent from System 3 to System 2.
    """
    model_config = ConfigDict(extra="ignore")

    request_type: str = Field(default="score_goals")
    period_id: int
    total_goals: int
    goals: List[GoalRecord]


# -----------------------------
# Payload received: System 2 -> System 3
# -----------------------------

class GoalScore(BaseModel):
    """
    Scored result per goal coming back from System 2.
    """
    model_config = ConfigDict(extra="ignore")

    goal_id: int
    period_id: int

    attainability_score: Optional[float] = None
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None
    integrity_score: Optional[float] = None
    overall_coherence_score: Optional[float] = None

    confidence: Optional[float] = None
    explanation: Optional[str] = None


class System2OutputPayload(BaseModel):
    """
    Full scored response returned by System 2.
    """
    model_config = ConfigDict(extra="ignore")

    request_type: str = Field(default="score_goals_result")
    period_id: int
    total_goals: int
    scores: List[GoalScore]


# -----------------------------
# Dashboard output: System 3 -> System 1
# -----------------------------

class DashboardGoalResult(BaseModel):
    """
    Clean dashboard-ready row sent back to System 1.
    """
    model_config = ConfigDict(extra="ignore")

    goal_id: int
    period_id: int

    goal_name: Optional[str] = None
    metric_name: Optional[str] = None
    metric_unit: Optional[str] = None
    bucket_name: Optional[str] = None
    parent_bucket_name: Optional[str] = None
    scenario_story: Optional[str] = None

    status_band: Optional[str] = None
    underfunded_flag: Optional[bool] = None
    overfunded_flag: Optional[bool] = None

    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    target_value_final_period: Optional[float] = None

    allocated_amount: Optional[float] = None
    allocation_efficiency_ratio: Optional[float] = None
    probability_of_hitting_target: Optional[float] = None
    time_to_green_estimate: Optional[float] = None

    attainability_score: Optional[float] = None
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None
    integrity_score: Optional[float] = None
    overall_coherence_score: Optional[float] = None

    confidence: Optional[float] = None
    explanation: Optional[str] = None


class System3ToSystem1Payload(BaseModel):
    """
    Final payload returned from System 3 to System 1 for dashboard display.
    """
    model_config = ConfigDict(extra="ignore")

    period_id: int
    total_goals: int
    dashboard_data: List[DashboardGoalResult]
    metadata: Optional[Dict[str, Any]] = None
