"""
Drug Feature Extractor
"""

import logging
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from utils.database import DrugDatabase

logger = logging.getLogger(__name__)


class DrugFeatureExtractor:

    def __init__(self):
        self.db = DrugDatabase()
        self.scaler = StandardScaler()
        self._stop_tokens = {
            "and", "with", "tablet", "tablets", "capsule", "capsules", "mg",
            "sodium", "hydrochloride", "extended", "release", "oral", "drug",
            "solution", "injection", "usp", "dose", "low"
        }
        logger.info("Initialized DrugFeatureExtractor")

    def _tokenize(self, text, min_len=3):
        raw = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return {t for t in raw if len(t) >= min_len and t not in self._stop_tokens}

    def _jaccard(self, a, b):
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _term_hits(self, text, terms):
        haystack = str(text or "").lower()
        hits = 0
        for term in sorted(terms, key=len, reverse=True):
            if len(term) < 4:
                continue
            if re.search(rf"\b{re.escape(term)}\b", haystack):
                hits += 1
        return hits

    def _risk_profile_flags(self, info):
        full = " ".join([
            str(info.get("generic", "")),
            str(info.get("active_ingredient", "")),
            str(info.get("category", "")),
            str(info.get("warnings", "")),
        ]).lower()
        anticoagulant = any(k in full for k in [
            "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "heparin", "enoxaparin"
        ])
        antiplatelet = any(k in full for k in [
            "aspirin", "clopidogrel", "prasugrel", "ticagrelor"
        ])
        nsaid = any(k in full for k in [
            "ibuprofen", "naproxen", "ketorolac", "indomethacin", "diclofenac", "meloxicam", "celecoxib"
        ])
        bleeding_word = "bleeding" in full or "hemorrhage" in full
        return anticoagulant, antiplatelet, nsaid, bleeding_word

    def extract_features(self, drug1, drug2):

        info1 = self.db.get_drug_info(drug1)
        info2 = self.db.get_drug_info(drug2)

        if not info1 or not info2:
            return None

        features = {}
        rule_out = self.db.check_interaction(drug1, drug2)
        if isinstance(rule_out, dict):
            features["rule_signal"] = 1
        else:
            features["rule_signal"] = int(rule_out) if rule_out is not None else 0
        active1 = self._tokenize(info1["active_ingredient"])
        active2 = self._tokenize(info2["active_ingredient"])
        generic1 = self._tokenize(info1["generic"])
        generic2 = self._tokenize(info2["generic"])
        class1 = self._tokenize(info1["category"])
        class2 = self._tokenize(info2["category"])
        warn1 = self._tokenize(info1["warnings"])
        warn2 = self._tokenize(info2["warnings"])
        contra1 = self._tokenize(info1["contraindications"])
        contra2 = self._tokenize(info2["contraindications"])

        t2 = self._tokenize(" ".join([drug2, info2["generic"], info2["active_ingredient"]]))
        t1 = self._tokenize(" ".join([drug1, info1["generic"], info1["active_ingredient"]]))
        text12 = str(info1["drug_interactions_text"]).lower()
        text21 = str(info2["drug_interactions_text"]).lower()

        # Same active ingredient
        features["same_ingredient"] = int(
            info1["active_ingredient"] != "" and
            info1["active_ingredient"] == info2["active_ingredient"]
        )

        features["same_category"] = int(info1["category"] != "" and info1["category"] == info2["category"])
        features["ingredient_jaccard"] = self._jaccard(active1, active2)
        features["generic_jaccard"] = self._jaccard(generic1, generic2)
        features["class_jaccard"] = self._jaccard(class1, class2)

        # Mentions in interaction narratives (directional).
        hits12 = self._term_hits(text12, t2)
        hits21 = self._term_hits(text21, t1)
        features["interaction_hits_forward"] = hits12
        features["interaction_hits_reverse"] = hits21
        features["interaction_keyword"] = int(hits12 > 0 or hits21 > 0)

        # Safety text overlap
        features["warnings_overlap"] = self._jaccard(warn1, warn2)
        features["contra_overlap"] = self._jaccard(contra1, contra2)

        # Relative use text size difference (scale-friendly)
        u1 = max(len(info1["uses"]), 1)
        u2 = max(len(info2["uses"]), 1)
        features["uses_length_ratio"] = min(u1, u2) / max(u1, u2)

        a1, p1, n1, b1 = self._risk_profile_flags(info1)
        a2, p2, n2, b2 = self._risk_profile_flags(info2)
        # Handcrafted high-risk combination prior.
        features["bleeding_combo_flag"] = int(
            ((a1 and (n2 or p2)) or (a2 and (n1 or p1))) and (b1 or b2)
        )

        return features

    def extract_pair_features(self, drug1, drug2, db=None):
        # Accept external db for app compatibility.
        if db is not None:
            self.db = db
        features = self.extract_features(drug1, drug2)
        if not features:
            return np.zeros(13, dtype=float)
        return np.asarray(list(features.values()), dtype=float)

    def build_dataset(self):

        X = []
        y = []

        interactions = self.db.get_interactions()

        for drug1, drug2, label in interactions:
            features = self.extract_features(drug1, drug2)

            if features:
                X.append(list(features.values()))
                y.append(label)

        logger.info(f"Feature matrix shape: ({len(X)}, {len(X[0]) if X else 0})")

        return X, y

    def normalize_features(self, X, fit=False):
        if isinstance(X, pd.DataFrame):
            arr = X.values.astype(float)
        else:
            arr = np.asarray(X, dtype=float)
        if fit:
            return self.scaler.fit_transform(arr)
        return self.scaler.transform(arr)


class DatasetBuilder:
    """Compatibility wrapper used by train.py."""

    def __init__(self, db=None):
        self.db = db if db is not None else DrugDatabase()
        self.extractor = DrugFeatureExtractor()
        # Ensure extractor uses the same DB instance as the builder.
        self.extractor.db = self.db

    def build_interaction_dataset(self):
        X, y = self.extractor.build_dataset()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        return X, y
