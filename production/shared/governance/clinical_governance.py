"""
Clinical Governance Overlay
===========================

Integrates clinical validation into the experimental component lifecycle.
Ensures all therapeutic features meet clinical safety and effectiveness standards.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

class RiskLevel(Enum):
    """Clinical risk levels for experimental components"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ClinicalChecklist:
    """Clinical governance checklist for component evaluation"""

    therapeutic_rationale: str
    risk_level: RiskLevel
    psychological_risks: List[str]
    consent_handling_paths: List[str]
    data_privacy_impact: str
    clinical_supervisor_approval: str
    monitoring_requirements: Optional[Dict[str, Any]] = None
    ethical_considerations: Optional[List[str]] = None
    patient_safety_measures: Optional[List[str]] = None

    def __post_init__(self):
        if self.monitoring_requirements is None:
            self.monitoring_requirements = {}
        if self.ethical_considerations is None:
            self.ethical_considerations = []
        if self.patient_safety_measures is None:
            self.patient_safety_measures = []

    def validate_checklist(self) -> bool:
        """Validate that all clinical requirements are met"""
        required_fields = [
            self.therapeutic_rationale,
            self.clinical_supervisor_approval
        ]

        if not all(required_fields):
            return False

        # Risk-based validation
        if self.risk_level == RiskLevel.HIGH:
            if not self.monitoring_requirements.get("intensive_supervision"):
                return False
            if not self.patient_safety_measures:
                return False

        if self.risk_level == RiskLevel.CRITICAL:
            if not self.monitoring_requirements.get("clinical_trial_required"):
                return False

        return True

    def generate_clinical_report(self) -> Dict[str, Any]:
        """Generate clinical validation report"""
        return {
            "therapeutic_impact": {
                "rationale": self.therapeutic_rationale,
                "expected_benefits": self.monitoring_requirements.get("expected_benefits", []),
                "potential_harms": self.psychological_risks
            },
            "risk_assessment": {
                "level": self.risk_level.value,
                "identified_risks": self.psychological_risks,
                "mitigation_strategies": self.patient_safety_measures,
                "monitoring_plan": self.monitoring_requirements
            },
            "privacy_compliance": {
                "consent_handling": self.consent_handling_paths,
                "data_impact": self.data_privacy_impact
            },
            "ethical_compliance": {
                "considerations": self.ethical_considerations,
                "supervisor_approval": self.clinical_supervisor_approval
            },
            "validation_date": datetime.now().isoformat()
        }

class ClinicalSupervisor:
    """Represents clinical supervision and approval authority"""

    def __init__(self, name: str, credentials: str, specializations: List[str]):
        self.name = name
        self.credentials = credentials
        self.specializations = specializations
        self.approval_log = []

    def approve_component(self, component_name: str, risk_level: RiskLevel,
                         rationale: str) -> Dict[str, Any]:
        """Approve a component for clinical use"""
        approval = {
            "component": component_name,
            "supervisor": self.name,
            "credentials": self.credentials,
            "risk_level": risk_level.value,
            "approval_date": datetime.now().isoformat(),
            "rationale": rationale,
            "conditions": self.get_approval_conditions(risk_level)
        }

        self.approval_log.append(approval)
        return approval

    def get_approval_conditions(self, risk_level: RiskLevel) -> List[str]:
        """Get approval conditions based on risk level"""
        conditions = []

        if risk_level == RiskLevel.LOW:
            conditions = ["Standard monitoring required"]
        elif risk_level == RiskLevel.MEDIUM:
            conditions = [
                "Weekly clinical review required",
                "Patient feedback monitoring",
                "Rapid rollback capability"
            ]
        elif risk_level == RiskLevel.HIGH:
            conditions = [
                "Daily clinical supervision required",
                "Real-time safety monitoring",
                "Immediate rollback capability",
                "Patient consent verification"
            ]
        elif risk_level == RiskLevel.CRITICAL:
            conditions = [
                "Continuous clinical oversight required",
                "Independent safety review board approval",
                "Clinical trial protocols required",
                "Emergency intervention protocols"
            ]

        return conditions

class ClinicalGovernanceManager:
    """Manages clinical governance for experimental components"""

    def __init__(self):
        self.supervisors = self.load_supervisors()
        self.review_log = []
        self.approval_matrix = self.create_approval_matrix()

    def load_supervisors(self) -> Dict[str, ClinicalSupervisor]:
        """Load clinical supervisors (in production, this would be from a database)"""
        return {
            "dr_sarah_chen": ClinicalSupervisor(
                name="Dr. Sarah Chen",
                credentials="PhD Clinical Psychology, Board Certified",
                specializations=["Crisis Intervention", "Therapeutic AI", "Digital Mental Health"]
            ),
            "dr_michael_torres": ClinicalSupervisor(
                name="Dr. Michael Torres",
                credentials="MD Psychiatry, Clinical Informatics",
                specializations=["Personalization", "AI Ethics", "Therapeutic Alliance"]
            ),
            "dr_emily_rodriguez": ClinicalSupervisor(
                name="Dr. Emily Rodriguez",
                credentials="PhD Health Informatics, Clinical Ethics",
                specializations=["Safety Protocols", "Regulatory Compliance", "Risk Assessment"]
            )
        }

    def create_approval_matrix(self) -> Dict[str, List[str]]:
        """Create approval matrix based on risk levels and component types"""
        return {
            "crisis_related": ["dr_sarah_chen", "dr_emily_rodriguez"],
            "personalization": ["dr_michael_torres", "dr_sarah_chen"],
            "safety_systems": ["dr_emily_rodriguez", "dr_sarah_chen"],
            "voice_features": ["dr_michael_torres"],  # Lower risk
            "ui_features": ["dr_michael_torres"]      # Lower risk
        }

    def require_clinical_review(self, component_name: str, risk_assessment: Dict[str, Any]) -> bool:
        """Determine if component requires clinical review"""
        high_risk_indicators = [
            "crisis" in component_name.lower(),
            "suicide" in component_name.lower(),
            "self_harm" in component_name.lower(),
            "diagnosis" in component_name.lower(),
            "treatment" in component_name.lower(),
            risk_assessment.get("risk_level") in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value],
            risk_assessment.get("affects_patient_safety", False),
            risk_assessment.get("changes_therapeutic_relationship", False)
        ]

        return any(high_risk_indicators)

    def schedule_clinical_review(self, component_name: str, risk_level: RiskLevel) -> Dict[str, Any]:
        """Schedule clinical review meeting"""
        required_supervisors = self.get_required_supervisors(component_name, risk_level)

        return {
            "component": component_name,
            "risk_level": risk_level.value,
            "required_supervisors": required_supervisors,
            "timeline": self.get_review_timeline(risk_level),
            "deliverables": [
                "Clinical safety assessment",
                "Therapeutic rationale validation",
                "Risk mitigation plan",
                "Monitoring protocol",
                "Approval decision"
            ],
            "escalation_criteria": self.get_escalation_criteria(risk_level)
        }

    def get_required_supervisors(self, component_name: str, risk_level: RiskLevel) -> List[str]:
        """Get required clinical supervisors for review"""
        # Check approval matrix first
        for category, supervisors in self.approval_matrix.items():
            if category in component_name.lower():
                return supervisors

        # Default based on risk level
        if risk_level == RiskLevel.CRITICAL:
            return ["dr_sarah_chen", "dr_michael_torres", "dr_emily_rodriguez"]
        elif risk_level == RiskLevel.HIGH:
            return ["dr_sarah_chen", "dr_emily_rodriguez"]
        elif risk_level == RiskLevel.MEDIUM:
            return ["dr_michael_torres", "dr_sarah_chen"]
        else:
            return ["dr_michael_torres"]

    def get_review_timeline(self, risk_level: RiskLevel) -> str:
        """Get review timeline based on risk level"""
        if risk_level == RiskLevel.CRITICAL:
            return "Within 24 hours"
        elif risk_level == RiskLevel.HIGH:
            return "Within 3 business days"
        elif risk_level == RiskLevel.MEDIUM:
            return "Within 1 week"
        else:
            return "Within 2 weeks"

    def get_escalation_criteria(self, risk_level: RiskLevel) -> List[str]:
        """Get criteria for escalating the review"""
        if risk_level == RiskLevel.CRITICAL:
            return ["Any safety concern identified", "Clinical disagreement", "Ethical issue raised"]
        elif risk_level == RiskLevel.HIGH:
            return ["Multiple safety concerns", "Significant clinical disagreement"]
        else:
            return ["Major safety or ethical concern identified"]

    def validate_clinical_safety(self, component_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive clinical safety validation"""
        validation_results = {
            "psychological_safety": self.assess_psychological_safety(component_analysis),
            "therapeutic_effectiveness": self.assess_therapeutic_effectiveness(component_analysis),
            "ethical_compliance": self.assess_ethical_compliance(component_analysis),
            "patient_safety": self.assess_patient_safety(component_analysis),
            "overall_recommendation": "pending",
            "required_monitoring": [],
            "approval_conditions": []
        }

        # Determine overall recommendation
        assessments = [
            validation_results["psychological_safety"],
            validation_results["therapeutic_effectiveness"],
            validation_results["ethical_compliance"],
            validation_results["patient_safety"]
        ]

        if all(assessment["status"] == "approved" for assessment in assessments):
            validation_results["overall_recommendation"] = "approved"
        elif any(assessment["status"] == "rejected" for assessment in assessments):
            validation_results["overall_recommendation"] = "rejected"
        else:
            validation_results["overall_recommendation"] = "requires_modification"

        # Add monitoring and conditions based on risk level
        risk_level = component_analysis.get("risk_level", RiskLevel.LOW)
        if risk_level == RiskLevel.HIGH:
            validation_results["required_monitoring"].extend([
                "Weekly clinical review",
                "Patient outcome monitoring",
                "Adverse event tracking"
            ])
            validation_results["approval_conditions"].extend([
                "Clinical supervisor sign-off required",
                "Patient consent documentation",
                "Emergency rollback capability"
            ])

        return validation_results

    def assess_psychological_safety(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess psychological safety impact"""
        risks = analysis.get("psychological_risks", [])

        if "harm_self" in risks or "suicide" in risks:
            return {"status": "rejected", "reason": "High-risk psychological safety concerns"}

        if len(risks) > 2:
            return {"status": "requires_review", "reason": "Multiple psychological risk factors"}

        return {"status": "approved", "reason": "Acceptable psychological safety profile"}

    def assess_therapeutic_effectiveness(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess therapeutic effectiveness"""
        effectiveness_score = analysis.get("therapeutic_effectiveness", 0)

        if effectiveness_score >= 4.2:
            return {"status": "approved", "reason": f"Strong therapeutic effectiveness: {effectiveness_score}/5.0"}
        elif effectiveness_score >= 3.5:
            return {"status": "conditional", "reason": f"Moderate effectiveness: {effectiveness_score}/5.0"}
        else:
            return {"status": "rejected", "reason": f"Insufficient therapeutic effectiveness: {effectiveness_score}/5.0"}

    def assess_ethical_compliance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess ethical compliance"""
        ethical_issues = analysis.get("ethical_considerations", [])

        if "privacy_violation" in ethical_issues or "consent_issue" in ethical_issues:
            return {"status": "rejected", "reason": "Critical ethical compliance issues"}

        if len(ethical_issues) > 0:
            return {"status": "requires_review", "reason": f"Ethical considerations identified: {ethical_issues}"}

        return {"status": "approved", "reason": "No significant ethical concerns"}

    def assess_patient_safety(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess patient safety impact"""
        safety_measures = analysis.get("patient_safety_measures", [])

        if not safety_measures:
            return {"status": "rejected", "reason": "No patient safety measures specified"}

        if len(safety_measures) >= 3:
            return {"status": "approved", "reason": f"Comprehensive safety measures: {len(safety_measures)} protocols"}

        return {"status": "requires_review", "reason": f"Limited safety measures: {len(safety_measures)} protocols"}

    def generate_governance_report(self) -> Dict[str, Any]:
        """Generate comprehensive clinical governance report"""
        return {
            "total_reviews": len(self.review_log),
            "supervisors_active": len(self.supervisors),
            "approval_rate": self.calculate_approval_rate(),
            "risk_distribution": self.analyze_risk_distribution(),
            "common_concerns": self.identify_common_concerns(),
            "compliance_status": "excellent" if self.calculate_approval_rate() > 0.95 else "good"
        }

    def calculate_approval_rate(self) -> float:
        """Calculate clinical approval rate"""
        if not self.review_log:
            return 1.0

        approved = sum(1 for review in self.review_log if review.get("recommendation") == "approved")
        return approved / len(self.review_log)

    def analyze_risk_distribution(self) -> Dict[str, int]:
        """Analyze distribution of risk levels in reviews"""
        distribution = {level.value: 0 for level in RiskLevel}

        for review in self.review_log:
            risk_level = review.get("risk_level", RiskLevel.LOW.value)
            distribution[risk_level] += 1

        return distribution

    def identify_common_concerns(self) -> List[str]:
        """Identify most common clinical concerns"""
        concerns = []
        for review in self.review_log:
            if "concerns" in review:
                concerns.extend(review["concerns"])

        # Return most common (simplified)
        return list(set(concerns))[:5]

# Global clinical governance manager
_clinical_governance = ClinicalGovernanceManager()

def get_clinical_governance_manager() -> ClinicalGovernanceManager:
    """Get the global clinical governance manager"""
    return _clinical_governance
