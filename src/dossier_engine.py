# src/dossier_engine.py
from pydantic import BaseModel, Field
from typing import List, Union

class FeatureEvidenceItem(BaseModel):
    signal: str
    value: str
    weight: str = None
    interpretation: str = None

class EvidenceDossier(BaseModel):
    ticket_id: str
    assigned_priority: str
    inferred_severity: str
    mismatch_type: str  # "Hidden Crisis" or "False Alarm"
    severity_delta: int
    feature_evidence: List[FeatureEvidenceItem]
    constraint_analysis: str = Field(..., max_length=350)
    confidence: str

class DossierEngine:
    @staticmethod
    def extract_escalation_keywords(text: str) -> str:
        """
        Scans description for critical action/escalation phrases.
        """
        keywords = ["immediately", "broken", "fail", "crash", "error", "prevent", "block", "loss", "urgently", "severe"]
        found = [w for w in keywords if w in text.lower()]
        return f"Keywords detected: {', '.join(found)}" if found else "No escalation keywords flagged."

    @classmethod
    def generate(cls, row: dict, predicted_mismatch: int, confidence_score: float) -> dict:
        """
        Compiles structural fields into a schema-compliant Evidence Dossier.
        """
        p_assigned_str = row['Ticket Priority']
        s_inf_str = row['inferred_severity_label']
        
        p_assigned = row['P_assigned']
        s_inf = row['inferred_severity']
        
        # Determine mismatch categorization
        mismatch_type = "Hidden Crisis" if s_inf > p_assigned else "False Alarm"
        severity_delta = abs(int(s_inf) - int(p_assigned))
        
        # Build strict feature evidence items traceable directly to source columns
        evidence = [
            FeatureEvidenceItem(
                signal="keyword",
                value=cls.extract_escalation_keywords(row['Ticket Description']),
                weight="High" if s_inf > 2 else "Medium"
            ),
            FeatureEvidenceItem(
                signal="resolution_time",
                value=f"{row['Resolution Time']} hours",
                interpretation=f"Ticket took {row['resolution_z_score']:.2f} standard deviations from the category median."
            )
        ]
        
        # Construct factual explanation grounded strictly to input features
        analysis = (
            f"Ticket ID {row['Ticket ID']} was logged via {row['Ticket Channel']} under category {row['Ticket Type']}. "
            f"The customer tier was flagged via domain '{row['Customer Domain']}'. The ticket was assigned a priority "
            f"of '{p_assigned_str}', but operational latency and text indicators match a severity profile of '{s_inf_str}'."
        )
        
        dossier = EvidenceDossier(
            ticket_id=str(row['Ticket ID']),
            assigned_priority=p_assigned_str,
            inferred_severity=s_inf_str,
            mismatch_type=mismatch_type,
            severity_delta=severity_delta,
            feature_evidence=evidence,
            constraint_analysis=analysis,
            confidence=f"{confidence_score:.2%}"
        )
        
        return dossier.model_dump()