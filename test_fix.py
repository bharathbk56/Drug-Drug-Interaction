"""
FINAL DRUG-DRUG INTERACTION PIPELINE
Rule-Based Labeling + ML Training
"""

import os
import pandas as pd
import random
from utils.openfda_loader import OpenFDALoader
from utils.preprocessing import DrugFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# -------------------------------------------------------
# STEP 1: DOWNLOAD FDA DATA
# -------------------------------------------------------

print("\n[Step 1/6] Downloading FDA Data...\n")

loader = OpenFDALoader()

drug_list = [
    "Aspirin", "Ibuprofen", "Paracetamol", "Metformin",
    "Warfarin", "Clopidogrel", "Omeprazole", "Prednisone",
    "Naproxen", "Atenolol", "Glipizide", "Furosemide",
    "Ramipril", "Meloxicam", "Pravastatin",
    "Verapamil", "Lisinopril", "Methotrexate",
    "Clindamycin", "Acetaminophen", "Itraconazole",
    "Ketorolac", "Indomethacin", "Metronidazole",
    "Ezetimibe", "Simvastatin", "Fluvastatin"
]

df = loader.batch_download_drugs(drug_list)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/fda_drugs.csv", index=False)

print(f"Downloaded {len(df)} drugs")


# -------------------------------------------------------
# STEP 2: SAVE FULL DATABASE
# -------------------------------------------------------

print("\n[Step 2/6] Saving processed database...\n")

os.makedirs("data/processed", exist_ok=True)

df = df.fillna("")

# Required column for DrugDatabase
df["drug_name"] = (
    df["brand_name"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df.to_csv("data/processed/drugs_database.csv", index=False)

print("Saved processed database.")


# -------------------------------------------------------
# STEP 3: RULE-BASED INTERACTION GENERATION
# -------------------------------------------------------

print("\n[Step 3/6] Generating rule-based interaction dataset...\n")

interaction_pairs = []
drug_names = df["drug_name"].tolist()

for i in range(len(drug_names)):
    for j in range(i + 1, len(drug_names)):

        d1 = drug_names[i]
        d2 = drug_names[j]

        ingredient1 = str(
            df.loc[df["drug_name"] == d1, "active_ingredient"].values[0]
        ).lower()

        ingredient2 = str(
            df.loc[df["drug_name"] == d2, "active_ingredient"].values[0]
        ).lower()

        class1 = str(
            df.loc[df["drug_name"] == d1, "pharm_class_epc"].values[0]
        ).lower()

        class2 = str(
            df.loc[df["drug_name"] == d2, "pharm_class_epc"].values[0]
        ).lower()

        label = 0

        # Rule 1: Same active ingredient
        if ingredient1 and ingredient1 == ingredient2:
            label = 1

        # Rule 2: Same pharmacologic class
        elif class1 and class2 and class1 == class2:
            label = 1

        # Rule 3: Known risky combinations
        elif ("warfarin" in d1 and "naproxen" in d2) or \
             ("warfarin" in d2 and "naproxen" in d1):
            label = 1

        elif ("lisinopril" in d1 and "naproxen" in d2) or \
             ("lisinopril" in d2 and "naproxen" in d1):
            label = 1

        interaction_pairs.append({
            "drug1": d1,
            "drug2": d2,
            "label": label
        })

interaction_df = pd.DataFrame(interaction_pairs)


# -------------------------------------------------------
# BALANCE DATASET
# -------------------------------------------------------

positive = interaction_df[interaction_df["label"] == 1]
negative = interaction_df[interaction_df["label"] == 0]

if len(positive) > 0:
    negative = negative.sample(len(positive) * 2, random_state=42)
    interaction_df = pd.concat([positive, negative])
    interaction_df = interaction_df.sample(frac=1, random_state=42)

interaction_df.to_csv("data/processed/drug_interactions.csv", index=False)

print("Saved balanced interaction dataset.")
print(interaction_df["label"].value_counts())


# -------------------------------------------------------
# STEP 4: BUILD ML DATASET
# -------------------------------------------------------

print("\n[Step 4/6] Building ML dataset...\n")

extractor = DrugFeatureExtractor()
X, y = extractor.build_dataset()

if len(set(y)) < 2:
    raise ValueError("Still not enough class diversity.")

print(f"Feature matrix size: {len(X)} samples")


# -------------------------------------------------------
# STEP 5: TRAIN MODEL
# -------------------------------------------------------

print("\n[Step 5/6] Training ML model...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------------
# STEP 6: SAVE MODEL
# -------------------------------------------------------

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/drug_interaction_model.pkl")

print("\nModel saved to models/drug_interaction_model.pkl")
print("\nPipeline Complete 🚀")