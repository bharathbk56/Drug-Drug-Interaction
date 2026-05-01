"""
OpenFDA API Integration (FINAL STABLE VERSION)
Fixes:
- Missing text
- Incomplete labels
- HTML tags
- Multiple label merging
- Section merging
- Synonym support
"""

import requests
import time
import logging
import re
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenFDALoader:

    BASE_URL = "https://api.fda.gov/drug"

    def __init__(self):
        self.session = requests.Session()
        self.rate_limit_delay = 0.25
        logger.info("Initialized Improved OpenFDA API Loader")

    # ------------------------------------------------------------
    # API REQUEST
    # ------------------------------------------------------------
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params, timeout=20)

            if response.status_code == 200:
                return response.json()

            logger.warning(f"API error {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    # ------------------------------------------------------------
    # CLEANERS
    # ------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"<.*?>", "", str(text))  # Remove HTML
        return " ".join(text.split())

    def _merge_list_field(self, data: Dict, key: str) -> str:
        value = data.get(key)
        if not value:
            return ""

        if isinstance(value, list):
            return " ".join(self._clean_text(v) for v in value)

        return self._clean_text(value)

    def _merge_openfda_field(self, openfda: Dict, key: str) -> str:
        value = openfda.get(key)
        if not value:
            return ""

        if isinstance(value, list):
            return ", ".join(value)

        return str(value)

    # ------------------------------------------------------------
    # SEARCH DRUG
    # ------------------------------------------------------------
    def search_drug_by_name(self, drug_name: str) -> Optional[Dict]:

        # Synonym mapping
        synonyms = {
            "paracetamol": "acetaminophen"
        }

        drug_name = synonyms.get(drug_name.lower(), drug_name)

        params = {
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": 5  # fetch multiple labels
        }

        data = self._make_request("label.json", params)

        if not data or "results" not in data:
            return None

        merged_result = {}

        # Merge multiple label entries
        for result in data["results"]:
            parsed = self._parse_drug_label(result)

            for key, value in parsed.items():
                if value:
                    merged_result[key] = (
                        merged_result.get(key, "") + " " + value
                    ).strip()

        return merged_result if merged_result else None

    # ------------------------------------------------------------
    # PARSE LABEL
    # ------------------------------------------------------------
    def _parse_drug_label(self, label_data: Dict) -> Dict:

        openfda = label_data.get("openfda", {})

        return {

            # Basic
            "brand_name": self._merge_openfda_field(openfda, "brand_name"),
            "generic_name": self._merge_openfda_field(openfda, "generic_name"),
            "manufacturer": self._merge_openfda_field(openfda, "manufacturer_name"),
            "product_type": self._merge_openfda_field(openfda, "product_type"),
            "route": self._merge_openfda_field(openfda, "route"),

            # Identifiers
            "rxcui": self._merge_openfda_field(openfda, "rxcui"),
            "spl_set_id": self._merge_openfda_field(openfda, "spl_set_id"),
            "substance_name": self._merge_openfda_field(openfda, "substance_name"),
            "unii": self._merge_openfda_field(openfda, "unii"),
            "active_ingredient": self._merge_openfda_field(openfda, "substance_name"),

            # Pharmacologic
            "pharm_class_epc": self._merge_openfda_field(openfda, "pharm_class_epc"),
            "pharm_class_cs": self._merge_openfda_field(openfda, "pharm_class_cs"),
            "pharm_class_moa": self._merge_openfda_field(openfda, "pharm_class_moa"),

            # Clinical Sections (merged properly)
            "indications": self._merge_list_field(label_data, "indications_and_usage"),
            "dosage": self._merge_list_field(label_data, "dosage_and_administration"),
            "contraindications": self._merge_list_field(label_data, "contraindications"),

            "warnings": (
                self._merge_list_field(label_data, "warnings") + " " +
                self._merge_list_field(label_data, "warnings_and_cautions")
            ).strip(),

            "adverse_reactions": (
                self._merge_list_field(label_data, "adverse_reactions") + " " +
                self._merge_list_field(label_data, "adverse_reactions_table")
            ).strip(),

            "boxed_warning": self._merge_list_field(label_data, "boxed_warning"),

            # Interactions
            "drug_interactions": self._merge_list_field(label_data, "drug_interactions"),

            # Advanced
            "clinical_pharmacology": self._merge_list_field(label_data, "clinical_pharmacology"),
            "mechanism_of_action": self._merge_list_field(label_data, "mechanism_of_action"),
            "pharmacokinetics": self._merge_list_field(label_data, "pharmacokinetics"),
            "overdosage": self._merge_list_field(label_data, "overdosage"),
            "description": self._merge_list_field(label_data, "description"),
            "how_supplied": self._merge_list_field(label_data, "how_supplied"),
            "storage_and_handling": self._merge_list_field(label_data, "storage_and_handling"),
            "abuse": self._merge_list_field(label_data, "abuse"),
            "dependence": self._merge_list_field(label_data, "dependence"),
            "nonclinical_toxicology": self._merge_list_field(label_data, "nonclinical_toxicology"),

            # Special Populations
            "pregnancy": self._merge_list_field(label_data, "pregnancy"),
            "pediatric_use": self._merge_list_field(label_data, "pediatric_use"),
            "geriatric_use": self._merge_list_field(label_data, "geriatric_use"),
        }

    # ------------------------------------------------------------
    # BATCH DOWNLOAD
    # ------------------------------------------------------------
    def batch_download_drugs(self, drug_names: List[str]) -> pd.DataFrame:
        drugs_data = []

        logger.info(f"Downloading {len(drug_names)} drugs...")

        for i, name in enumerate(drug_names, 1):
            logger.info(f"[{i}/{len(drug_names)}] {name}")
            data = self.search_drug_by_name(name)

            if data:
                drugs_data.append(data)

        return pd.DataFrame(drugs_data)

    # ------------------------------------------------------------
    # CSV SAVE
    # ------------------------------------------------------------
    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False)
        logger.info(f"Saved to {filepath}")