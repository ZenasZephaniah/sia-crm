# src/dossier_engine.py
"""
Stage 3 of the Support Integrity Auditor (SIA) pipeline.

The DossierEngine generates a structured, hallucination-free Evidence
Dossier for every ticket flagged as a priority mismatch. The output
schema is enforced by the Pydantic EvidenceDossier model and matches
the JSON schema required by the project spec.

Hallucination discipline:
  - Every value in feature_evidence is computed from a specific input
    field on the ticket row (no LLM generation, no inference beyond
    the classifier's predicted mismatch flag).
  - constraint_analysis is built from a deterministic template that
    interpolates only fields present on the input row.
  - All interpolated string values are truncated to a safe length
    before insertion so the result never exceeds the Pydantic
    max_length=500 constraint on constraint_analysis.
  - If for any reason a row cannot be rendered into a valid dossier,
    a safe fallback dossier is returned (with an explanatory
    constraint_analysis) instead of raising, so that a single
    pathological row cannot crash a batch inference run.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureEvidenceItem(BaseModel):
    signal: str
    value: str
    weight: Optional[str] = None
    interpretation: Optional[str] = None


class EvidenceDossier(BaseModel):
    ticket_id: str
    assigned_priority: str
    inferred_severity: str
    mismatch_type: str  # "Hidden Crisis" or "False Alarm"
    severity_delta: int
    feature_evidence: List[FeatureEvidenceItem]
    constraint_analysis: str = Field(..., max_length=500)
    confidence: str


# Maximum length for any single interpolated field in constraint_analysis.
# Leaves headroom under the 500-char Pydantic limit after the fixed
# template text (~200 chars) and the other interpolated fields.
_FIELD_MAX = 60


def _truncate(value, limit: int = _FIELD_MAX) -> str:
    """Safely stringify and truncate to a max length."""
    s = str(value) if value is not None else ""
    if len(s) > limit:
        return s[:limit]
    return s


class DossierEngine:
    @staticmethod
    def extract_escalation_keywords(text: str) -> str:
        """Scan description for critical action/escalation phrases."""
        keywords = [
            "immediately", "broken", "fail", "crash", "error",
            "prevent", "block", "loss", "urgently", "severe",
        ]
        found = [w for w in keywords if w in text.lower()]
        return (
            f"Keywords detected: {', '.join(found)}"
            if found
            else "No escalation keywords flagged."
        )

    @classmethod
    def generate(cls, row: dict, predicted_mismatch: int, confidence_score: float) -> dict:
        """
        Compile a deterministic Evidence Dossier for a single ticket row.

        Parameters
        ----------
        row : dict
            A dictionary representation of a single ticket. Must contain
            at least: 'Ticket Priority', 'inferred_severity_label',
            'P_assigned', 'inferred_severity', 'Ticket Description',
            'Resolution Time', 'resolution_z_score', 'Ticket ID',
            'Ticket Channel', 'Ticket Type', 'Customer Domain'.
        predicted_mismatch : int
            The binary classifier's prediction (0 or 1). Currently
            informational only; the dossier is generated for any row
            passed in (callers gate on predicted_mismatch == 1).
        confidence_score : float
            The classifier's confidence for its prediction, in [0, 1].

        Returns
        -------
        dict
            A JSON-serializable dict matching the EvidenceDossier schema.
        """
        try:
            # --- Compute mismatch type and severity delta (all numeric
            # operations protected against bad input) ---
            try:
                p_assigned = int(row['P_assigned'])
            except (KeyError, TypeError, ValueError):
                p_assigned = 0
            try:
                s_inf = int(row['inferred_severity'])
            except (KeyError, TypeError, ValueError):
                s_inf = 0

            mismatch_type = "Hidden Crisis" if s_inf > p_assigned else "False Alarm"
            severity_delta = abs(s_inf - p_assigned)

            # --- Build evidence items (all values traceable to input row) ---
            evidence = [
                FeatureEvidenceItem(
                    signal="keyword",
                    value=cls.extract_escalation_keywords(
                        row.get('Ticket Description', '')
                    ),
                    weight="High" if s_inf > 2 else "Medium",
                ),
                FeatureEvidenceItem(
                    signal="resolution_time",
                    value=f"{row.get('Resolution Time', 'unknown')} hours",
                    interpretation=(
                        f"Ticket took {row.get('resolution_z_score', 0.0):.2f} "
                        f"standard deviations from the category median."
                    ),
                ),
            ]

            # --- Build constraint_analysis with all values truncated ---
            ticket_id = _truncate(row.get('Ticket ID', 'UNKNOWN'), 30)
            channel = _truncate(row.get('Ticket Channel', 'unknown'), _FIELD_MAX)
            ticket_type = _truncate(row.get('Ticket Type', 'unknown'), _FIELD_MAX)
            domain = _truncate(row.get('Customer Domain', 'unknown'), _FIELD_MAX)
            priority = _truncate(row.get('Ticket Priority', 'unknown'), 20)
            severity_label = _truncate(row.get('inferred_severity_label', 'unknown'), 20)

            analysis = (
                f"Ticket {ticket_id} was logged via {channel} under category "
                f"{ticket_type}. The customer tier was flagged via domain "
                f"'{domain}'. The ticket was assigned a priority of "
                f"'{priority}', but operational latency and text indicators "
                f"match a severity profile of '{severity_label}'."
            )

            # --- Final defensive truncation in case any field was longer
            # than expected; leaves 20 chars of margin under 500 ---
            if len(analysis) > 480:
                analysis = analysis[:480]

            dossier = EvidenceDossier(
                ticket_id=str(row.get('Ticket ID', 'UNKNOWN')),
                assigned_priority=str(row.get('Ticket Priority', 'unknown')),
                inferred_severity=str(row.get('inferred_severity_label', 'unknown')),
                mismatch_type=mismatch_type,
                severity_delta=severity_delta,
                feature_evidence=evidence,
                constraint_analysis=analysis,
                confidence=f"{confidence_score:.2%}",
            )
            return dossier.model_dump()

        except Exception as e:
            # Defensive fallback: never let one bad row crash a batch.
            # Returns a valid dossier with an explanatory analysis.
            fallback_analysis = (
                f"Dossier generation encountered an issue for this ticket: "
                f"{str(e)[:200]}. Underlying input fields may be missing or malformed."
            )
            try:
                fallback = EvidenceDossier(
                    ticket_id=str(row.get('Ticket ID', 'UNKNOWN')),
                    assigned_priority=str(row.get('Ticket Priority', 'unknown')),
                    inferred_severity=str(row.get('inferred_severity_label', 'unknown')),
                    mismatch_type="Hidden Crisis" if predicted_mismatch else "False Alarm",
                    severity_delta=0,
                    feature_evidence=[
                        FeatureEvidenceItem(
                            signal="keyword",
                            value="Could not extract keywords due to input error.",
                            weight="Unknown",
                        ),
                        FeatureEvidenceItem(
                            signal="resolution_time",
                            value="unknown",
                            interpretation="Could not compute z-score due to input error.",
                        ),
                    ],
                    constraint_analysis=fallback_analysis,
                    confidence=f"{confidence_score:.2%}",
                )
                return fallback.model_dump()
            except Exception:
                # Absolute last resort: hand-build a dict matching the schema
                return {
                    "ticket_id": str(row.get('Ticket ID', 'UNKNOWN')),
                    "assigned_priority": "unknown",
                    "inferred_severity": "unknown",
                    "mismatch_type": "Hidden Crisis" if predicted_mismatch else "False Alarm",
                    "severity_delta": 0,
                    "feature_evidence": [],
                    "constraint_analysis": "Dossier generation failed; see logs.",
                    "confidence": f"{confidence_score:.2%}",
                }
