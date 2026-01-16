"""
OpenEMR API Client

Provides integration with OpenEMR's REST and FHIR APIs for retrieving
patient data to provide context to the LLM.

Supports:
- OAuth2 authentication with PKCE
- FHIR R4 API for patient data
- Standard REST API fallback
- Patient demographics, conditions, medications, allergies, etc.

Based on OpenEMR API documentation:
https://github.com/openemr/openemr/blob/master/API_README.md
"""

import time
import base64
import hashlib
import secrets
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode

import requests
import structlog

logger = structlog.get_logger(__name__)


class OpenEMRClient:
    """Client for OpenEMR FHIR and REST APIs."""

    def __init__(self, settings):
        """Initialize the OpenEMR client."""
        self.settings = settings
        self.base_url = settings.openemr_base_url.rstrip("/")
        self.api_path = settings.openemr_api_path
        self.fhir_path = settings.openemr_fhir_path
        self.client_id = settings.openemr_client_id
        self.client_secret = settings.openemr_client_secret
        self.redirect_uri = settings.openemr_redirect_uri
        self.scopes = settings.openemr_scopes_list
        self.verify_ssl = settings.openemr_verify_ssl

        self._access_token = None
        self._token_expires_at = 0
        self._refresh_token = None

        self.timeout = 30

        logger.info(
            "openemr_client_initialized",
            base_url=self.base_url,
            scopes=self.scopes,
        )

    @property
    def api_url(self) -> str:
        """Get full API URL."""
        return f"{self.base_url}{self.api_path}"

    @property
    def fhir_url(self) -> str:
        """Get full FHIR URL."""
        return f"{self.base_url}{self.fhir_path}"

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/fhir+json",
        }

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return

        if self._refresh_token:
            try:
                self._refresh_access_token()
                return
            except Exception as e:
                logger.warning("token_refresh_failed", error=str(e))

        # Need to authenticate
        self._authenticate()

    def _authenticate(self) -> None:
        """
        Authenticate with OpenEMR using OAuth2 Client Credentials flow.

        For server-to-server communication without user interaction.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "OpenEMR client_id and client_secret must be configured"
            )

        token_url = f"{self.base_url}/oauth2/default/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": " ".join(self.scopes),
        }

        try:
            response = requests.post(
                token_url,
                data=data,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data["access_token"]
            self._token_expires_at = time.time() + token_data.get("expires_in", 3600)
            self._refresh_token = token_data.get("refresh_token")

            logger.info("openemr_authenticated")

        except requests.exceptions.HTTPError as e:
            logger.error(
                "openemr_auth_failed",
                status=e.response.status_code,
                response=e.response.text,
            )
            raise RuntimeError(f"OpenEMR authentication failed: {e.response.text}")

    def _refresh_access_token(self) -> None:
        """Refresh the access token."""
        token_url = f"{self.base_url}/oauth2/default/token"

        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": self._refresh_token,
        }

        response = requests.post(
            token_url,
            data=data,
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()
        token_data = response.json()

        self._access_token = token_data["access_token"]
        self._token_expires_at = time.time() + token_data.get("expires_in", 3600)
        self._refresh_token = token_data.get("refresh_token", self._refresh_token)

        logger.info("openemr_token_refreshed")

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient demographics using FHIR API.

        Args:
            patient_id: The patient's UUID or internal ID

        Returns:
            Patient FHIR resource
        """
        self._ensure_authenticated()

        url = f"{self.fhir_url}/Patient/{patient_id}"

        response = requests.get(
            url,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()

    def get_patient_conditions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's conditions/diagnoses."""
        self._ensure_authenticated()

        url = f"{self.fhir_url}/Condition"
        params = {"patient": patient_id}

        response = requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        bundle = response.json()
        return [entry["resource"] for entry in bundle.get("entry", [])]

    def get_patient_medications(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's medications."""
        self._ensure_authenticated()

        url = f"{self.fhir_url}/MedicationRequest"
        params = {"patient": patient_id}

        response = requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        bundle = response.json()
        return [entry["resource"] for entry in bundle.get("entry", [])]

    def get_patient_allergies(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's allergies."""
        self._ensure_authenticated()

        url = f"{self.fhir_url}/AllergyIntolerance"
        params = {"patient": patient_id}

        response = requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        bundle = response.json()
        return [entry["resource"] for entry in bundle.get("entry", [])]

    def get_patient_observations(
        self, patient_id: str, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get patient's observations (vitals, labs, etc.).

        Args:
            patient_id: Patient ID
            category: Optional category filter (vital-signs, laboratory, etc.)
        """
        self._ensure_authenticated()

        url = f"{self.fhir_url}/Observation"
        params = {"patient": patient_id}
        if category:
            params["category"] = category

        response = requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        bundle = response.json()
        return [entry["resource"] for entry in bundle.get("entry", [])]

    def get_patient_encounters(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's encounters."""
        self._ensure_authenticated()

        url = f"{self.fhir_url}/Encounter"
        params = {"patient": patient_id}

        response = requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        bundle = response.json()
        return [entry["resource"] for entry in bundle.get("entry", [])]

    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of patient data for LLM context.

        This aggregates data from multiple endpoints into a single
        structured summary suitable for providing context to the LLM.
        """
        self._ensure_authenticated()

        summary = {
            "patient_id": patient_id,
            "demographics": {},
            "conditions": [],
            "medications": [],
            "allergies": [],
            "vitals": [],
            "recent_encounters": [],
        }

        try:
            # Get patient demographics
            patient = self.get_patient(patient_id)
            summary["demographics"] = self._extract_demographics(patient)
        except Exception as e:
            logger.warning("failed_to_get_demographics", error=str(e))

        try:
            # Get conditions
            conditions = self.get_patient_conditions(patient_id)
            summary["conditions"] = [
                self._extract_condition(c) for c in conditions
            ]
        except Exception as e:
            logger.warning("failed_to_get_conditions", error=str(e))

        try:
            # Get medications
            medications = self.get_patient_medications(patient_id)
            summary["medications"] = [
                self._extract_medication(m) for m in medications
            ]
        except Exception as e:
            logger.warning("failed_to_get_medications", error=str(e))

        try:
            # Get allergies
            allergies = self.get_patient_allergies(patient_id)
            summary["allergies"] = [
                self._extract_allergy(a) for a in allergies
            ]
        except Exception as e:
            logger.warning("failed_to_get_allergies", error=str(e))

        try:
            # Get recent vitals
            vitals = self.get_patient_observations(patient_id, "vital-signs")
            summary["vitals"] = [
                self._extract_observation(v) for v in vitals[:10]
            ]
        except Exception as e:
            logger.warning("failed_to_get_vitals", error=str(e))

        try:
            # Get recent encounters
            encounters = self.get_patient_encounters(patient_id)
            summary["recent_encounters"] = [
                self._extract_encounter(e) for e in encounters[:5]
            ]
        except Exception as e:
            logger.warning("failed_to_get_encounters", error=str(e))

        logger.info(
            "patient_summary_retrieved",
            patient_id=patient_id,
            conditions_count=len(summary["conditions"]),
            medications_count=len(summary["medications"]),
        )

        return summary

    def _extract_demographics(self, patient: Dict) -> Dict[str, Any]:
        """Extract relevant demographics from FHIR Patient resource."""
        demographics = {}

        # Gender
        demographics["gender"] = patient.get("gender")

        # Birth date (will be anonymized later)
        demographics["birth_date"] = patient.get("birthDate")

        # Name (will be anonymized later)
        names = patient.get("name", [])
        if names:
            name = names[0]
            demographics["name"] = f"{' '.join(name.get('given', []))} {name.get('family', '')}"

        return demographics

    def _extract_condition(self, condition: Dict) -> Dict[str, Any]:
        """Extract relevant data from FHIR Condition resource."""
        return {
            "code": self._get_coding_display(condition.get("code")),
            "status": condition.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
            "onset": condition.get("onsetDateTime"),
            "category": self._get_coding_display(condition.get("category", [{}])[0]) if condition.get("category") else None,
        }

    def _extract_medication(self, medication: Dict) -> Dict[str, Any]:
        """Extract relevant data from FHIR MedicationRequest resource."""
        return {
            "medication": self._get_coding_display(medication.get("medicationCodeableConcept")),
            "status": medication.get("status"),
            "intent": medication.get("intent"),
            "dosage": medication.get("dosageInstruction", [{}])[0].get("text") if medication.get("dosageInstruction") else None,
        }

    def _extract_allergy(self, allergy: Dict) -> Dict[str, Any]:
        """Extract relevant data from FHIR AllergyIntolerance resource."""
        return {
            "substance": self._get_coding_display(allergy.get("code")),
            "type": allergy.get("type"),
            "category": allergy.get("category", []),
            "criticality": allergy.get("criticality"),
            "reaction": [
                {
                    "manifestation": self._get_coding_display(m)
                    for m in r.get("manifestation", [])
                }
                for r in allergy.get("reaction", [])
            ],
        }

    def _extract_observation(self, observation: Dict) -> Dict[str, Any]:
        """Extract relevant data from FHIR Observation resource."""
        result = {
            "code": self._get_coding_display(observation.get("code")),
            "status": observation.get("status"),
            "date": observation.get("effectiveDateTime"),
        }

        # Extract value
        if "valueQuantity" in observation:
            vq = observation["valueQuantity"]
            result["value"] = f"{vq.get('value')} {vq.get('unit', '')}"
        elif "valueString" in observation:
            result["value"] = observation["valueString"]
        elif "valueCodeableConcept" in observation:
            result["value"] = self._get_coding_display(observation["valueCodeableConcept"])

        return result

    def _extract_encounter(self, encounter: Dict) -> Dict[str, Any]:
        """Extract relevant data from FHIR Encounter resource."""
        return {
            "type": [self._get_coding_display(t) for t in encounter.get("type", [])],
            "status": encounter.get("status"),
            "class": encounter.get("class", {}).get("display"),
            "period": encounter.get("period"),
            "reason": [
                self._get_coding_display(r.get("concept"))
                for r in encounter.get("reason", [])
                if r.get("concept")
            ],
        }

    def _get_coding_display(self, codeable_concept: Optional[Dict]) -> Optional[str]:
        """Get display text from a CodeableConcept."""
        if not codeable_concept:
            return None

        # Try display text first
        if codeable_concept.get("text"):
            return codeable_concept["text"]

        # Try coding display
        codings = codeable_concept.get("coding", [])
        for coding in codings:
            if coding.get("display"):
                return coding["display"]

        # Fall back to code
        for coding in codings:
            if coding.get("code"):
                return coding["code"]

        return None

    def search_patients(
        self, name: Optional[str] = None, identifier: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for patients by name or identifier."""
        self._ensure_authenticated()

        url = f"{self.fhir_url}/Patient"
        params = {}

        if name:
            params["name"] = name
        if identifier:
            params["identifier"] = identifier

        response = requests.get(
            url,
            params=params,
            headers=self._get_headers(),
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        response.raise_for_status()

        bundle = response.json()
        return [entry["resource"] for entry in bundle.get("entry", [])]

    def health_check(self) -> bool:
        """Check if OpenEMR API is accessible."""
        try:
            url = f"{self.fhir_url}/metadata"
            response = requests.get(
                url,
                verify=self.verify_ssl,
                timeout=10,
            )
            return response.ok
        except Exception:
            return False


def format_patient_context(summary: Dict[str, Any]) -> str:
    """
    Format patient summary as text suitable for LLM context.

    This creates a human-readable summary that can be included
    in the LLM prompt.
    """
    lines = ["=== Patient Clinical Summary ===\n"]

    # Demographics
    demo = summary.get("demographics", {})
    if demo:
        lines.append(f"Gender: {demo.get('gender', 'Unknown')}")
        if demo.get("birth_date"):
            lines.append(f"DOB: {demo['birth_date']}")

    # Conditions
    conditions = summary.get("conditions", [])
    if conditions:
        lines.append("\n--- Active Conditions ---")
        for c in conditions:
            if c.get("code"):
                status = f" ({c['status']})" if c.get("status") else ""
                lines.append(f"- {c['code']}{status}")

    # Medications
    medications = summary.get("medications", [])
    if medications:
        lines.append("\n--- Current Medications ---")
        for m in medications:
            if m.get("medication"):
                dosage = f": {m['dosage']}" if m.get("dosage") else ""
                lines.append(f"- {m['medication']}{dosage}")

    # Allergies
    allergies = summary.get("allergies", [])
    if allergies:
        lines.append("\n--- Allergies ---")
        for a in allergies:
            if a.get("substance"):
                crit = f" (CRITICAL)" if a.get("criticality") == "high" else ""
                lines.append(f"- {a['substance']}{crit}")

    # Recent Vitals
    vitals = summary.get("vitals", [])
    if vitals:
        lines.append("\n--- Recent Vitals ---")
        for v in vitals[:5]:
            if v.get("code") and v.get("value"):
                lines.append(f"- {v['code']}: {v['value']}")

    lines.append("\n=== End Summary ===")

    return "\n".join(lines)
