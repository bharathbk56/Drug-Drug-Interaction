"""
Model Training Script
Train ML models for drug interaction prediction
(FULLY STABLE – SAFE FOR SMALL DATASETS)
"""

import numpy as np
import os
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Import custom modules
from models.drug_classifier import DrugInteractionClassifier
from models.interaction_predictor import InteractionPredictor
from utils.database import DrugDatabase
from utils.preprocessing import DatasetBuilder, DrugFeatureExtractor

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# SAFE EVALUATION (CRITICAL FIX)
# --------------------------------------------------
def safe_classification_report(y_true, y_pred):
    """
    Safe classification report for small / imbalanced datasets
    """
    labels = np.unique(y_true)
    target_names = [
        "No Interaction" if l == 0 else "Interaction"
        for l in labels
    ]

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )
    )


# --------------------------------------------------
# CONFIG LOADER
# --------------------------------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# RANDOM FOREST TRAINING
# --------------------------------------------------
def train_random_forest(config):
    logger.info("=" * 60)
    logger.info("TRAINING RANDOM FOREST CLASSIFIER")
    logger.info("=" * 60)

    db = DrugDatabase()
    builder = DatasetBuilder(db)

    logger.info("Building dataset...")
    X, y = builder.build_interaction_dataset()

    if len(X) < 5:
        raise ValueError("Dataset too small to train Random Forest")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["train_test_split"],
        random_state=config["data"]["random_seed"],
        stratify=y if len(np.unique(y)) > 1 else None
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=config["data"]["random_seed"],
        stratify=y_train if len(np.unique(y_train)) > 1 else None
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    classifier_cfg = dict(config["model"]["classifier"])
    model_type = classifier_cfg.pop("type", "random_forest")
    classifier = DrugInteractionClassifier(
        model_type=model_type,
        **classifier_cfg
    )

    classifier.train(X_fit, y_fit)
    val_proba = classifier.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_acc = -1.0
    for thr in np.linspace(0.30, 0.75, 46):
        val_pred = (val_proba >= thr).astype(int)
        acc = accuracy_score(y_val, val_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(thr)

    # Refit on full training split, then evaluate on held-out test split.
    classifier.train(X_train, y_train)
    probabilities = classifier.predict_proba(X_test)
    predictions = (probabilities[:, 1] >= best_threshold).astype(int)

    logger.info("\nMODEL EVALUATION")
    print("\nClassification Report:")
    safe_classification_report(y_test, predictions)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    logger.info(f"Selected decision threshold: {best_threshold:.2f}")

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probabilities[:, 1])
        logger.info(f"ROC AUC Score: {auc:.3f}")
    else:
        logger.warning("ROC AUC skipped (single class present)")

    os.makedirs("data/models", exist_ok=True)
    classifier.save_model("data/models/drug_classifier.pkl")

    logger.info("✅ Random Forest training completed!")
    return classifier


# --------------------------------------------------
# NEURAL NETWORK TRAINING
# --------------------------------------------------
def train_neural_network(config):
    logger.info("=" * 60)
    logger.info("TRAINING NEURAL NETWORK")
    logger.info("=" * 60)

    db = DrugDatabase()
    builder = DatasetBuilder(db)

    logger.info("Building dataset...")
    X, y = builder.build_interaction_dataset()

    if len(X) < 5:
        raise ValueError("Dataset too small to train Neural Network")

    extractor = DrugFeatureExtractor()
    X_norm = extractor.normalize_features(X, fit=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm,
        y,
        test_size=config["data"]["train_test_split"] + config["data"]["validation_split"],
        random_state=config["data"]["random_seed"],
        stratify=y if len(np.unique(y)) > 1 else None
    )

    val_ratio = config["data"]["validation_split"] / (
        config["data"]["train_test_split"] + config["data"]["validation_split"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=config["data"]["random_seed"]
    )

    logger.info(f"Training set: {X_train.shape[0]}")
    logger.info(f"Validation set: {X_val.shape[0]}")
    logger.info(f"Test set: {X_test.shape[0]}")

    predictor = InteractionPredictor(
        input_dim=X.shape[1],
        config=config["model"]["neural_network"]
    )

    predictor.train(X_train, y_train, X_val, y_val)

    predictions, probabilities = predictor.predict(X_test)

    logger.info("\nMODEL EVALUATION")
    print("\nClassification Report:")
    safe_classification_report(y_test, predictions)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probabilities)
        logger.info(f"ROC AUC Score: {auc:.3f}")
    else:
        logger.warning("ROC AUC skipped (single class present)")

    predictor.save_model("data/models/interaction_predictor.pth")
    logger.info("✅ Neural Network training completed!")
    return predictor


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.warning("config.yaml not found. Using default config.")
        config = {
            "model": {
                "classifier": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "random_state": 42
                },
                "neural_network": {
                    "hidden_layers": [256, 128, 64],
                    "dropout": 0.3,
                    "learning_rate": 0.001,
                    "epochs": 50,
                    "batch_size": 32
                }
            },
            "data": {
                "train_test_split": 0.2,
                "validation_split": 0.1,
                "random_seed": 42
            }
        }

    logger.info("🚀 Starting Model Training Pipeline")

    rf_ok = nn_ok = False

    try:
        train_random_forest(config)
        rf_ok = True
    except Exception as e:
        logger.error(f"Random Forest failed: {e}")

    try:
        train_neural_network(config)
        nn_ok = True
    except Exception as e:
        logger.error(f"Neural Network failed: {e}")

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Random Forest: {'✅ Success' if rf_ok else '❌ Failed'}")
    logger.info(f"Neural Network: {'✅ Success' if nn_ok else '❌ Failed'}")
    logger.info("Models saved in: data/models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
