"""
Security Utilities

Provides input sanitization, PHI anonymization, and other
security-related functions for HIPAA compliance.
"""

import re
import html
from typing import Dict, Any, Optional
import hashlib


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: Raw user input

    Returns:
        Sanitized text safe for processing
    """
    if not text:
        return ""

    # HTML entity encoding to prevent XSS
    text = html.escape(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Limit consecutive whitespace
    text = re.sub(r"\s{10,}", " " * 10, text)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


def anonymize_phi(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anonymize Protected Health Information (PHI) before sending to LLM.

    This implements de-identification according to HIPAA Safe Harbor method,
    removing or masking the 18 HIPAA identifiers.

    Args:
        patient_data: Dictionary containing patient information

    Returns:
        Anonymized patient data safe for LLM processing
    """
    if not patient_data:
        return {}

    # Create a copy to avoid modifying original
    anonymized = {}

    # HIPAA Safe Harbor: 18 identifiers to remove/mask
    phi_fields = {
        # Direct identifiers - remove entirely
        "name", "first_name", "last_name", "middle_name",
        "ssn", "social_security", "mrn", "medical_record_number",
        "email", "email_address",
        "phone", "phone_number", "telephone", "fax",
        "address", "street", "city", "state", "zip", "postal_code",
        "ip_address", "device_id",
        "license_number", "vehicle_id",
        "account_number", "certificate_number",
        "photo", "image", "biometric",

        # Indirect identifiers - mask or generalize
        "date_of_birth", "dob", "birthdate",
        "admission_date", "discharge_date",
        "death_date",
    }

    # Fields to keep but potentially generalize
    safe_fields = {
        "gender", "sex",
        "allergies", "allergy_list",
        "medications", "medication_list",
        "conditions", "diagnoses", "diagnosis_list",
        "vitals", "vital_signs",
        "lab_results", "labs",
        "procedures",
        "immunizations",
        "chief_complaint", "reason_for_visit",
        "symptoms",
        "notes", "clinical_notes",  # Should be reviewed manually
    }

    for key, value in patient_data.items():
        key_lower = key.lower().replace("-", "_").replace(" ", "_")

        if key_lower in phi_fields:
            # Handle dates - generalize to age range
            if "date" in key_lower or "dob" in key_lower or "birth" in key_lower:
                anonymized[key] = _generalize_date(value)
            else:
                # Remove other PHI
                anonymized[key] = "[REDACTED]"

        elif key_lower in safe_fields:
            # Keep safe clinical data
            anonymized[key] = value

        elif isinstance(value, dict):
            # Recursively anonymize nested dictionaries
            anonymized[key] = anonymize_phi(value)

        elif isinstance(value, list):
            # Handle lists
            anonymized[key] = [
                anonymize_phi(item) if isinstance(item, dict) else item
                for item in value
            ]

        else:
            # Unknown fields - check for PHI patterns
            anonymized[key] = _mask_potential_phi(str(value))

    return anonymized


def _generalize_date(date_value: Any) -> str:
    """Convert a date to an age range for anonymization."""
    if not date_value:
        return "[DATE REDACTED]"

    try:
        from datetime import datetime, date

        if isinstance(date_value, str):
            # Try common date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]:
                try:
                    parsed = datetime.strptime(date_value, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                return "[DATE REDACTED]"
        elif isinstance(date_value, (datetime, date)):
            parsed = date_value if isinstance(date_value, date) else date_value.date()
        else:
            return "[DATE REDACTED]"

        # Calculate age
        today = date.today()
        age = today.year - parsed.year - (
            (today.month, today.day) < (parsed.month, parsed.day)
        )

        # Return age range
        if age < 1:
            return "Age: <1 year"
        elif age < 18:
            return f"Age: {age // 5 * 5}-{age // 5 * 5 + 4} years (pediatric)"
        elif age < 65:
            return f"Age: {age // 10 * 10}-{age // 10 * 10 + 9} years (adult)"
        elif age < 90:
            return f"Age: {age // 10 * 10}-{age // 10 * 10 + 9} years (elderly)"
        else:
            return "Age: 90+ years"

    except Exception:
        return "[DATE REDACTED]"


def _mask_potential_phi(text: str) -> str:
    """Mask potential PHI patterns in free text."""
    if not text:
        return text

    # SSN pattern
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", text)

    # Phone numbers
    text = re.sub(
        r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[PHONE REDACTED]",
        text,
    )

    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[EMAIL REDACTED]",
        text,
    )

    # MRN patterns (common formats)
    text = re.sub(r"\b(?:MRN|MR#?|Medical Record)[\s:]*\d{6,10}\b", "[MRN REDACTED]", text, flags=re.IGNORECASE)

    # Dates in various formats (be careful not to over-match)
    # Only mask if appears to be a specific date
    text = re.sub(
        r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b",
        "[DATE REDACTED]",
        text,
    )

    return text


def generate_session_token() -> str:
    """Generate a secure session token."""
    import secrets
    return secrets.token_urlsafe(32)


def hash_for_audit(data: str) -> str:
    """Create a one-way hash for audit logging (can verify but not reverse)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def validate_patient_id(patient_id: str) -> bool:
    """Validate patient ID format."""
    if not patient_id:
        return False

    # Allow alphanumeric IDs with hyphens
    if not re.match(r"^[a-zA-Z0-9\-]{1,50}$", patient_id):
        return False

    return True


def rate_limit_key(request) -> str:
    """Generate a key for rate limiting based on request."""
    # Use IP + user agent hash for rate limiting
    ip = request.remote_addr or "unknown"
    ua = request.headers.get("User-Agent", "")
    return f"{ip}:{hash_for_audit(ua)}"
