"""
Drug Database Utility
(ML-safe + Streamlit-safe + Production-safe)
"""

import os
import pandas as pd
import logging
import random
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugDatabase:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        drug_db_path = os.path.join(
            base_dir, "data", "processed", "drugs_database.csv"
        )

        if not os.path.exists(drug_db_path):
            raise FileNotFoundError(
                f"Drug database not found: {drug_db_path}"
            )

        df = pd.read_csv(drug_db_path)
        df["drug_name"] = df["drug_name"].astype(str).str.lower().str.strip()

        self.drugs = {}

        def safe_str(val):
            if pd.isna(val) or val is None:
                return ""
            return str(val)

        for _, row in df.iterrows():
            name = row.get("drug_name")
            if not name:
                continue

            uses_text = safe_str(row.get("uses", row.get("indications", "")))
            side_effects_text = safe_str(
                row.get("side_effects", row.get("adverse_reactions", ""))
            )

            self.drugs[name] = {
                "generic": safe_str(row.get("generic_name", "")),
                "category": safe_str(row.get("category", row.get("pharm_class_epc", ""))),
                "active_ingredient": safe_str(row.get("active_ingredient", "")),
                "uses": uses_text,
                "side_effects": side_effects_text,
                "warnings": safe_str(row.get("warnings", "")),
                "contraindications": safe_str(row.get("contraindications", "")),
                "drug_interactions_text": safe_str(
                    row.get("drug_interactions_text", row.get("drug_interactions", ""))
                )
            }

        logger.info(f"Loaded {len(self.drugs)} drugs")

        # ------------------------------
        # Load interactions (if exist)
        # ------------------------------
        self.interactions = []

        interaction_path = os.path.join(
            base_dir, "data", "processed", "drug_interactions.csv"
        )

        if os.path.exists(interaction_path):
            interactions_df = pd.read_csv(interaction_path)

            for _, row in interactions_df.iterrows():
                d1 = self.normalize(row.get("drug1"))
                d2 = self.normalize(row.get("drug2"))
                label = int(row.get("label", 0))

                if d1 and d2:
                    self.interactions.append((d1, d2, label))

            logger.info(f"Loaded {len(self.interactions)} interactions")

        if len(self.interactions) == 0:
            logger.warning("No real interactions found — generating synthetic data")
            self._generate_synthetic_interactions()

    # ----------------------------------
    def normalize(self, name):
        if not name or pd.isna(name):
            return None
        return str(name).lower().strip()

    # ----------------------------------
    def resolve_drug_key(self, drug_name):
        key = self.normalize(drug_name)
        if not key:
            return None

        if key in self.drugs:
            return key

        for stored_name, info in self.drugs.items():
            if key in stored_name or stored_name in key:
                return stored_name
            generic = self.normalize(info.get("generic", ""))
            if generic and (key in generic or generic in key):
                return stored_name

        return None

    # ----------------------------------
    def get_drug_info(self, drug_name):
        key = self.resolve_drug_key(drug_name)
        return self.drugs.get(key) if key else None

    # ----------------------------------
    def get_drug_count(self):
        return len(self.drugs)

    # ----------------------------------
    def get_all_drug_names(self):
        return sorted(self.drugs.keys())

    # ----------------------------------
    def get_interactions(self):
        return self.interactions

    # ----------------------------------
    def _known_pair_details(self, d1, d2):
        pair = frozenset([self.normalize(d1), self.normalize(d2)])
        details_map = {
            frozenset(["warfarin", "aspirin"]): {
                "severity": "HIGH",
                "confidence": 0.97,
                "description": (
                    "Warfarin + Aspirin can significantly increase bleeding risk. "
                    "Warfarin reduces clotting factor activity (anticoagulant effect), while "
                    "Aspirin inhibits platelet aggregation. The combination may cause additive "
                    "antithrombotic effects and increase risk of GI or intracranial bleeding."
                ),
                "recommendation": (
                    "Use combination only with clinician supervision. Monitor INR and bleeding signs "
                    "(e.g., bruising, melena, hematuria, prolonged bleeding)."
                ),
                "evidence_type": "Mechanism-based and label-supported interaction",
                "mechanism_points": [
                    "Warfarin decreases vitamin K-dependent clotting factors (II, VII, IX, X).",
                    "Aspirin irreversibly inhibits platelet COX-1, reducing thromboxane A2 and aggregation.",
                    "Combined anticoagulant + antiplatelet effects produce additive hemostatic impairment."
                ],
                "clinical_risk_points": [
                    "Higher probability of major bleeding than either drug alone.",
                    "Risk includes gastrointestinal bleeding and, less commonly, intracranial hemorrhage.",
                    "Risk may further increase in elderly patients or those with prior ulcer disease."
                ],
                "monitoring_points": [
                    "Track INR more frequently after initiation or dose changes.",
                    "Watch for bleeding indicators: melena, hematuria, gum bleeding, unusual bruising.",
                    "Review concurrent drugs that also increase bleeding risk (e.g., NSAIDs, SSRIs)."
                ],
                "management_points": [
                    "Use only when benefit outweighs bleeding risk and indication is clear.",
                    "Consider gastroprotection strategies in high GI-risk patients per clinician judgment.",
                    "Educate patient to promptly report bleeding symptoms and avoid self-medication with OTC NSAIDs."
                ],
                "research_notes": [
                    "Classification reflects rule-based clinical knowledge plus curated label-style interaction evidence.",
                    "Confidence is heuristic and not a substitute for patient-specific causal inference."
                ]
            },
            frozenset(["warfarin", "ibuprofen"]): {
                "severity": "HIGH",
                "confidence": 0.95,
                "description": (
                    "Warfarin + Ibuprofen may elevate bleeding risk due to concurrent anticoagulant "
                    "effect and NSAID-related GI mucosal injury/platelet effects."
                ),
                "recommendation": "Avoid routine co-use when possible; monitor INR and bleeding risk if unavoidable."
            },
            frozenset(["warfarin", "naproxen"]): {
                "severity": "HIGH",
                "confidence": 0.95,
                "description": (
                    "Warfarin + Naproxen is a high-risk combination for bleeding, especially gastrointestinal bleeding."
                ),
                "recommendation": "Prefer non-NSAID alternatives when clinically appropriate."
            },
            frozenset(["warfarin", "ketorolac"]): {
                "severity": "HIGH",
                "confidence": 0.98,
                "description": (
                    "Warfarin + Ketorolac has substantial bleeding risk and is generally considered a poor combination."
                ),
                "recommendation": "Avoid combination unless no alternatives and under strict clinical supervision."
            },
            frozenset(["warfarin", "indomethacin"]): {
                "severity": "HIGH",
                "confidence": 0.95,
                "description": (
                    "Warfarin + Indomethacin may increase bleeding risk via additive antithrombotic/GI effects."
                ),
                "recommendation": "Use caution and monitor for bleeding complications."
            },
            frozenset(["warfarin", "clopidogrel"]): {
                "severity": "HIGH",
                "confidence": 0.96,
                "description": (
                    "Warfarin + Clopidogrel increases major bleeding risk due to combined anticoagulant and antiplatelet effects."
                ),
                "recommendation": "If dual therapy is clinically required, close monitoring is essential."
            },
            frozenset(["warfarin", "metronidazole"]): {
                "severity": "HIGH",
                "confidence": 0.96,
                "description": (
                    "Metronidazole can increase Warfarin exposure (CYP-related interaction), potentially elevating INR and bleeding risk."
                ),
                "recommendation": "Consider warfarin dose adjustment and frequent INR checks during coadministration."
            },
        }

        return details_map.get(pair)

    # ----------------------------------
    def check_interaction(self, drug1, drug2):
        d1 = self.resolve_drug_key(drug1)
        d2 = self.resolve_drug_key(drug2)

        if not d1 or not d2:
            return 0

        # Return rich clinical details first for curated high-risk pairs.
        known_high_risk = [
            ("warfarin", "aspirin"),
            ("warfarin", "ibuprofen"),
            ("warfarin", "naproxen"),
            ("warfarin", "ketorolac"),
            ("warfarin", "indomethacin"),
            ("warfarin", "clopidogrel"),
            ("warfarin", "metronidazole"),
        ]
        d1n = self.normalize(d1)
        d2n = self.normalize(d2)
        for a, b in known_high_risk:
            if (a in d1n and b in d2n) or (a in d2n and b in d1n):
                details = self._known_pair_details(a, b)
                if details:
                    return details

        matched_label = None
        for a, b, label in self.interactions:
            if (a == d1 and b == d2) or (a == d2 and b == d1):
                # Keep match but allow downstream clinical overrides to upgrade risk.
                matched_label = int(label)
                if matched_label == 1:
                    return 1

        # Fallback: check FDA interaction narrative text in either direction.
        info1 = self.drugs.get(d1, {})
        info2 = self.drugs.get(d2, {})
        text1 = str(info1.get("drug_interactions_text", "")).lower()
        text2 = str(info2.get("drug_interactions_text", "")).lower()

        names2 = [
            self.normalize(drug2),
            self.normalize(info2.get("generic", "")),
            d2,
        ]
        names1 = [
            self.normalize(drug1),
            self.normalize(info1.get("generic", "")),
            d1,
        ]

        def has_term(text, terms):
            for term in terms:
                if not term:
                    continue
                token = term.split(",")[0].strip()
                if len(token) < 4:
                    continue
                if re.search(rf"\b{re.escape(token)}\b", text):
                    return True
            return False

        if has_term(text1, names2) or has_term(text2, names1):
            return 1

        # Conservative clinical overrides for high-risk combinations.
        for a, b in known_high_risk:
            if (a in d1n and b in d2n) or (a in d2n and b in d1n):
                details = self._known_pair_details(a, b)
                if details:
                    return details
                return {
                    "severity": "HIGH",
                    "confidence": 0.95,
                    "description": "Known clinically significant interaction based on curated high-risk rules.",
                    "recommendation": "Consult healthcare provider before co-administration."
                }

        if matched_label is not None:
            return matched_label

        return 0

    # ----------------------------------
    def has_duplicate_ingredient(self, drug1, drug2):
        info1 = self.get_drug_info(drug1)
        info2 = self.get_drug_info(drug2)
        if not info1 or not info2:
            return False

        ingredient1 = str(info1.get("active_ingredient", "")).strip().lower()
        ingredient2 = str(info2.get("active_ingredient", "")).strip().lower()

        return bool(ingredient1 and ingredient2 and ingredient1 == ingredient2)

    # ----------------------------------
    def _generate_synthetic_interactions(self, n=1000):
        drug_list = list(self.drugs.keys())

        for _ in range(n):
            d1, d2 = random.sample(drug_list, 2)
            label = 1 if random.random() < 0.2 else 0
            self.interactions.append((d1, d2, label))

        logger.info(f"Generated {len(self.interactions)} synthetic interactions")
