import joblib
import logging
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugInteractionClassifier:
    """
    Random Forest model wrapper for Drug–Drug Interaction prediction
    """

    def __init__(
        self,
        model_type="random_forest",
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ):
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1
            )
        else:
            raise ValueError("Unsupported model type")

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, X, y):
        logger.info("Training model...")
        self.model.fit(X, y)

    # -----------------------------
    # Inference
    # -----------------------------
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # -----------------------------
    # Utilities
    # -----------------------------
    def get_feature_importance(self):
        return getattr(self.model, "feature_importances_", None)

    def save_model(self, path):
        joblib.dump(self.model, path)
        logger.info(f"Model saved at: {path}")

    def load_model(self, path):
        """
        Load a trained model from disk
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from: {path}")
