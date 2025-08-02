from enum import IntEnum

class GDMRiskLevel(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

class ActionType(IntEnum):
    ROUTINE_MONITORING = 0
    INCREASED_MONITORING = 1
    GLUCOSE_TOLERANCE_TEST = 2
    CONTINUOUS_GLUCOSE_MONITORING = 3
    DIETARY_COUNSELING = 4
    EXERCISE_PROGRAM = 5
    WEIGHT_MANAGEMENT = 6
    STRESS_REDUCTION = 7
    INSULIN_THERAPY = 8
    METFORMIN_PRESCRIPTION = 9
    SPECIALIST_REFERRAL = 10
    IMMEDIATE_INTERVENTION = 11
    COMPREHENSIVE_ASSESSMENT = 12
    FAMILY_HISTORY_REVIEW = 13
    NO_ACTION = 14

def get_risk_level(gestational_week, blood_sugar=None):
    """Determine risk level based on gestational week and optional blood sugar."""
    if gestational_week < 20:
        return GDMRiskLevel.LOW
    elif gestational_week < 30:
        return GDMRiskLevel.MODERATE
    elif gestational_week < 35:
        return GDMRiskLevel.HIGH
    else:
        return GDMRiskLevel.CRITICAL
