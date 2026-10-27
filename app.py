"""
FairShift: Autonomous Shift Equity & Fatigue-Proofing Engine
Target: UN Sustainable Development Goal 8 (Decent Work & Economic Growth)
Framework: FastAPI (Production ASGI) + Pydantic v2
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Domain Models & Schemas
# -----------------------------------------------------------------------------

class Worker(BaseModel):
    id: str = Field(..., description="Unique worker identifier")
    name: str = Field(..., description="Worker full name")
    role: str = Field(..., description="Primary designation or competency")
    skill_level: int = Field(ge=1, le=5, description="Skill capability rating (1-5)")
    weekly_hours_logged: float = Field(ge=0.0, default=0.0, description="Hours logged in current rolling workweek")
    fatigue_index: float = Field(ge=0.0, le=1.0, default=0.0, description="Fatigue rating (0.0 to 1.0)")
    hourly_rate_inr: float = Field(ge=0.0, default=200.0, description="Base statutory hourly wage in INR")
    is_rest_compliant: bool = Field(default=True, description="Complies with mandatory minimum rest period")


class Shift(BaseModel):
    id: str = Field(..., description="Unique shift identifier")
    title: str = Field(..., description="Shift role designation")
    required_skill_level: int = Field(ge=1, le=5, description="Minimum skill required (1-5)")
    duration_hours: float = Field(gt=0, description="Duration of shift in hours")
    base_rate_inr: float = Field(gt=0, description="Shift base hourly rate in INR")
    hazard_level: str = Field(default="Low", description="Hazard categorization: Low, Medium, High")
    hazard_multiplier: float = Field(default=1.0, ge=1.0, description="Hazard allowance multiplier")


class EvaluationRequest(BaseModel):
    worker: Worker
    shift: Shift


class EvaluationResult(BaseModel):
    is_compliant: bool
    safety_score: int
    projected_hours: float
    hazard_lockout: bool
    flags: List[str]
    recommendation: str


class BatchAllocationRequest(BaseModel):
    workers: List[Worker]
    shifts: List[Shift]


class AssignmentRecord(BaseModel):
    shift_id: str
    shift_title: str
    worker_id: str
    worker_name: str
    assigned_hours: float
    effective_hourly_wage: float
    total_payout_inr: float
    post_shift_fatigue_index: float
    safety_score: int


class AllocationReport(BaseModel):
    timestamp_utc: str
    allocated_shifts: List[AssignmentRecord]
    unfilled_shifts: List[Dict[str, Any]]
    gini_inequality_coefficient: float
    total_wages_distributed_inr: float
    sdg_compliance_score: float


# -----------------------------------------------------------------------------
# Core SDG 8 Compliance & Optimization Engine
# -----------------------------------------------------------------------------

STATUTORY_WEEKLY_LIMIT = 48.0  # ILO Convention & Factories Act 1948 maximum weekly cap
BURNOUT_FATIGUE_THRESHOLD = 0.70  # Lockout point for high strain assignments


def compute_gini_coefficient(earnings: List[float]) -> float:
    """Calculates the Gini coefficient to measure shift wage equity (0 = total equality)."""
    valid = [e for e in earnings if e > 0]
    if not valid or len(valid) <= 1:
        return 0.0
    sorted_earnings = sorted(valid)
    n = len(sorted_earnings)
    cumulative_sum = sum((idx + 1) * val for idx, val in enumerate(sorted_earnings))
    total = sum(sorted_earnings)
    gini = (2.0 * cumulative_sum) / (n * total) - (n + 1.0) / n
    return round(max(0.0, min(gini, 1.0)), 3)


def evaluate_worker_shift_fit(worker: Worker, shift: Shift) -> EvaluationResult:
    """
    Evaluates shift safety against:
    1. Statutory overtime caps (48 hours/week limit).
    2. Dynamic fatigue levels and high-hazard lockout.
    3. Mandatory recovery rest compliance.
    """
    flags: List[str] = []
    safety_score = 100
    is_compliant = True
    hazard_lockout = False

    projected_hours = worker.weekly_hours_logged + shift.duration_hours

    # Rule 1: Weekly Cap Protection
    if projected_hours > STATUTORY_WEEKLY_LIMIT:
        is_compliant = False
        excess = round(projected_hours - STATUTORY_WEEKLY_LIMIT, 1)
        flags.append(f"Statutory limit breached: {projected_hours}h exceeds the 48h/week threshold by {excess}h.")
        safety_score -= 45

    # Rule 2: Minimum Rest Window
    if not worker.is_rest_compliant:
        is_compliant = False
        flags.append("Mandatory 11-hour inter-shift recovery window violated.")
        safety_score -= 30

    # Rule 3: Fatigue Index & Hazard Level Lockout
    if worker.fatigue_index > BURNOUT_FATIGUE_THRESHOLD:
        safety_score -= 25
        flags.append(f"High fatigue detected ({int(worker.fatigue_index * 100)}%). Worker is at elevated burnout risk.")
        if shift.hazard_level.lower() in ["medium", "high"]:
            is_compliant = False
            hazard_lockout = True
            flags.append("Hazard Lockout: Worker cannot operate in high/medium hazard zones while fatigued.")

    # Rule 4: Skill Rating Qualification
    if worker.skill_level < shift.required_skill_level:
        is_compliant = False
        flags.append(f"Competency shortfall: Worker skill ({worker.skill_level}) below requirement ({shift.required_skill_level}).")
        safety_score -= 20

    # Assignment recommendation strategy
    if not is_compliant:
        recommendation = "Reject - Non-compliant with SDG 8 decent work safety constraints."
    elif worker.weekly_hours_logged < 25.0:
        recommendation = "Priority Allocation - Promotes fair income balancing for under-allocated staff."
    elif worker.weekly_hours_logged > 40.0:
        recommendation = "Low Priority - Worker nearing safe weekly operational envelope."
    else:
        recommendation = "Balanced Allocation - Standard safe operational assignment."

    return EvaluationResult(
        is_compliant=is_compliant,
        safety_score=max(0, safety_score),
        projected_hours=round(projected_hours, 2),
        hazard_lockout=hazard_lockout,
        flags=flags,
        recommendation=recommendation
    )


# -----------------------------------------------------------------------------
# Application Setup & Endpoints
# -----------------------------------------------------------------------------

app = FastAPI(
    title="FairShift SDG 8 Compliance API",
    description="Intelligent Fair Labor Scheduling, Wage Equity, and Anti-Burnout Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health Check"])
def root() -> Dict[str, Any]:
    return {
        "service": "FairShift AI Decision Engine",
        "standard": "UN SDG Goal 8 - Decent Work & Economic Growth",
        "version": "1.0.0",
        "statutory_weekly_limit_hours": STATUTORY_WEEKLY_LIMIT,
        "burnout_threshold": BURNOUT_FATIGUE_THRESHOLD
    }


@app.post("/api/v1/evaluate", response_model=EvaluationResult, tags=["Verification"])
def evaluate_single_assignment(request: EvaluationRequest):
    """Verifies a single shift-to-worker allocation against labor safety standards."""
    return evaluate_worker_shift_fit(request.worker, request.shift)


@app.post("/api/v1/optimize-schedule", response_model=AllocationReport, tags=["Scheduler"])
def optimize_shifts(payload: BatchAllocationRequest):
    """
    Optimizes schedule distribution across candidate workers:
    - Eliminates discriminatory scheduling bias.
    - Balances hours logged to compress income inequality (Gini optimization).
    - Prevents hazardous fatigue overload.
    """
    if not payload.workers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Worker pool cannot be empty.")
    if not payload.shifts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Shift list cannot be empty.")

    worker_pool = {w.id: w.model_copy() for w in payload.workers}
    worker_payouts: Dict[str, float] = {w.id: 0.0 for w in payload.workers}
    allocations: List[AssignmentRecord] = []
    unfilled: List[Dict[str, Any]] = []

    # Sort shifts by complexity (higher skill requirements assigned first)
    sorted_shifts = sorted(payload.shifts, key=lambda s: s.required_skill_level, reverse=True)

    for shift in sorted_shifts:
        eligible_candidates = []

        for worker in worker_pool.values():
            eval_res = evaluate_worker_shift_fit(worker, shift)
            if eval_res.is_compliant:
                eligible_candidates.append((worker, eval_res))

        if not eligible_candidates:
            unfilled.append({
                "shift_id": shift.id,
                "shift_title": shift.title,
                "reason": "No compliant or qualified worker available under safety limits."
            })
            continue

        # Fairness heuristic: Prioritize lowest logged hours, then highest safety score
        eligible_candidates.sort(key=lambda item: (item[0].weekly_hours_logged, -item[1].safety_score))
        selected_worker, evaluation = eligible_candidates[0]

        # Calculate fair wage with applicable hazard multiplier
        base_rate = max(selected_worker.hourly_rate_inr, shift.base_rate_inr)
        effective_rate = round(base_rate * shift.hazard_multiplier, 2)
        total_payout = round(effective_rate * shift.duration_hours, 2)

        # Update running worker metrics
        selected_worker.weekly_hours_logged += shift.duration_hours
        fatigue_increment = shift.duration_hours * (0.04 if shift.hazard_level.lower() == "low" else 0.07)
        selected_worker.fatigue_index = min(1.0, round(selected_worker.fatigue_index + fatigue_increment, 2))
        worker_payouts[selected_worker.id] += total_payout

        allocations.append(AssignmentRecord(
            shift_id=shift.id,
            shift_title=shift.title,
            worker_id=selected_worker.id,
            worker_name=selected_worker.name,
            assigned_hours=shift.duration_hours,
            effective_hourly_wage=effective_rate,
            total_payout_inr=total_payout,
            post_shift_fatigue_index=selected_worker.fatigue_index,
            safety_score=evaluation.safety_score
        ))

    all_wages = list(worker_payouts.values())
    gini = compute_gini_coefficient(all_wages)

    # Compute overall SDG 8 compliance score (0 - 100)
    unfilled_penalty = (len(unfilled) / max(len(payload.shifts), 1)) * 30.0
    inequality_penalty = gini * 40.0
    compliance_score = max(0.0, round(100.0 - unfilled_penalty - inequality_penalty, 2))

    return AllocationReport(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        allocated_shifts=allocations,
        unfilled_shifts=unfilled,
        gini_inequality_coefficient=gini,
        total_wages_distributed_inr=round(sum(all_wages), 2),
        sdg_compliance_score=compliance_score
    )


# -----------------------------------------------------------------------------
# Local Development Runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
