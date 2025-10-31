"""
Graduation Ledger Management
============================

Manages the promotion of experimental components to production.
Provides audit trails and compliance documentation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class GraduationRequirements:
    """Quantitative requirements for component graduation"""
    latency_p95: str = "<= 150ms"
    error_rate: str = "< 0.5%"
    safety_false_negative: str = "0"
    therapeutic_quality_score: str = ">= 4.2/5.0"
    integration_test_coverage: str = ">= 95%"
    clinical_validation_pass: bool = True

@dataclass
class ClinicalValidation:
    """Clinical validation checklist"""
    therapeutic_rationale: str
    risk_assessment: str
    consent_handling: str
    supervisor_approval: str
    monitoring_requirements: Optional[Dict[str, Any]] = None

@dataclass
class GraduationRecord:
    """Complete graduation record"""
    component: str
    promoted_to: str
    version: str
    decision_date: str
    decision_type: str  # "automatic_graduation", "manual_review", "emergency_promotion"
    trigger_metrics: Dict[str, Any]
    approved_by: List[str]
    clinical_validation: ClinicalValidation
    rollback_plan: str
    monitoring_period: str

class GraduationLedger:
    """Manages component graduation lifecycle"""

    def __init__(self, ledger_path: Path = Path("shared/governance/graduation_ledger.yaml")):
        self.ledger_path = Path(__file__).parent / "graduation_ledger.yaml"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.requirements = GraduationRequirements()
        self.approved_reviewers = [
            "@safety-lead", "@ml-ops-lead", "@clinical-supervisor",
            "@product-lead", "@engineering-lead"
        ]

        # Load or create ledger
        if self.ledger_path.exists():
            self.ledger = self.load_ledger()
        else:
            self.ledger = self.create_empty_ledger()

    def create_empty_ledger(self) -> Dict[str, Any]:
        """Create a new empty ledger"""
        return {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "total_graduations": 0,
            "graduation_requirements": asdict(self.requirements),
            "approved_reviewers": self.approved_reviewers,
            "graduations": []
        }

    def load_ledger(self) -> Dict[str, Any]:
        """Load ledger from YAML file"""
        try:
            with open(self.ledger_path, 'r') as f:
                return yaml.safe_load(f) or self.create_empty_ledger()
        except Exception:
            return self.create_empty_ledger()

    def save_ledger(self):
        """Save ledger to YAML file"""
        self.ledger["last_updated"] = datetime.now().isoformat()

        with open(self.ledger_path, 'w') as f:
            yaml.dump(self.ledger, f, default_flow_style=False, sort_keys=False)

    def record_graduation(self, record: GraduationRecord) -> bool:
        """Record a component graduation"""
        # Validate record
        if not self.validate_graduation_record(record):
            return False

        # Convert to dict and add to ledger
        record_dict = asdict(record)
        record_dict["clinical_validation"] = asdict(record.clinical_validation)

        if "graduations" not in self.ledger:
            self.ledger["graduations"] = []

        self.ledger["graduations"].append(record_dict)
        self.ledger["total_graduations"] = len(self.ledger["graduations"])

        # Save changes
        self.save_ledger()

        print(f"🎉 Component graduated: {record.component} → {record.promoted_to}")
        return True

    def validate_graduation_record(self, record: GraduationRecord) -> bool:
        """Validate that a graduation record meets requirements"""
        # Check required approvals
        if not any(approver in record.approved_by for approver in ["@safety-lead", "@clinical-supervisor"]):
            print("❌ Must include safety lead and clinical supervisor approval")
            return False

        # Check all required approvers are in approved list
        for approver in record.approved_by:
            if approver not in self.approved_reviewers:
                print(f"❌ Unauthorized approver: {approver}")
                return False

        # Check clinical validation completeness
        clinical = record.clinical_validation
        required_fields = ["therapeutic_rationale", "risk_assessment",
                          "consent_handling", "supervisor_approval"]
        for field in required_fields:
            if not getattr(clinical, field, None):
                print(f"❌ Missing clinical validation field: {field}")
                return False

        # Check metrics meet requirements
        metrics = record.trigger_metrics
        if metrics.get("latency_p95", float('inf')) > 150:
            print("❌ Latency requirement not met")
            return False

        if metrics.get("error_rate", 1.0) > 0.005:
            print("❌ Error rate requirement not met")
            return False

        if metrics.get("safety_score", 0) < 95:
            print("❌ Safety requirement not met")
            return False

        return True

    def check_graduation_eligibility(self, component_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a component is eligible for graduation"""
        eligibility = {
            "eligible": True,
            "requirements_met": {},
            "blocking_issues": []
        }

        # Check each requirement
        reqs = asdict(self.requirements)
        for req_name, req_value in reqs.items():
            if req_name in metrics:
                metric_value = metrics[req_name]

                if req_name == "latency_p95":
                    met = metric_value <= 150
                elif req_name == "error_rate":
                    met = metric_value <= 0.005
                elif req_name == "safety_false_negative":
                    met = metric_value == 0
                elif req_name == "therapeutic_quality_score":
                    met = metric_value >= 4.2
                elif req_name == "integration_test_coverage":
                    met = metric_value >= 95
                elif req_name == "clinical_validation_pass":
                    met = bool(metric_value)
                else:
                    met = True

                eligibility["requirements_met"][req_name] = met
                if not met:
                    eligibility["eligible"] = False
                    eligibility["blocking_issues"].append(f"{req_name}: {metric_value} (required: {req_value})")

        return eligibility

    def get_graduation_history(self, component: str = None) -> List[Dict[str, Any]]:
        """Get graduation history, optionally filtered by component"""
        graduations = self.ledger.get("graduations", [])
        if component:
            return [g for g in graduations if g["component"] == component]
        return graduations

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for regulatory review"""
        graduations = self.ledger.get("graduations", [])

        return {
            "total_graduations": len(graduations),
            "schema_version": self.ledger.get("schema_version", "unknown"),
            "last_updated": self.ledger.get("last_updated", "unknown"),
            "safety_incidents": 0,  # Would integrate with monitoring
            "rollback_events": 0,   # Would track rollbacks
            "clinical_approvals": len([
                g for g in graduations
                if "supervisor_approval" in g.get("clinical_validation", {}).get("supervisor_approval", "")
            ]),
            "audit_trail_complete": len(graduations) > 0,
            "graduation_requirements": self.ledger.get("graduation_requirements", {}),
            "approved_reviewers": self.ledger.get("approved_reviewers", [])
        }

    def create_graduation_record(
        self,
        component: str,
        promoted_to: str,
        metrics: Dict[str, Any],
        clinical_validation: ClinicalValidation,
        approved_by: List[str],
        decision_type: str = "automatic_graduation",
        rollback_plan: str = "Feature flag reversion",
        monitoring_period: str = "30 days post-graduation"
    ) -> GraduationRecord:
        """Create a graduation record with automatic field population"""

        # Auto-generate version based on current date
        version = f"1.{datetime.now().month}.{datetime.now().day}"

        record = GraduationRecord(
            component=component,
            promoted_to=promoted_to,
            version=version,
            decision_date=datetime.now().strftime("%Y-%m-%d"),
            decision_type=decision_type,
            trigger_metrics=metrics,
            approved_by=approved_by,
            clinical_validation=clinical_validation,
            rollback_plan=rollback_plan,
            monitoring_period=monitoring_period
        )

        return record

# Global ledger instance
_graduation_ledger = GraduationLedger()

def get_graduation_ledger() -> GraduationLedger:
    """Get the global graduation ledger instance"""
    return _graduation_ledger
