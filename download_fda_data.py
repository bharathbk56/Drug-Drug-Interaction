"""
FINAL STABLE PIPELINE
1. Download FDA Data
2. Save full attributes
3. Generate improved interaction dataset
4. Balance dataset
5. Train ML model
"""

import os
import pandas as pd
import re
import json
import numpy as np
from utils.openfda_loader import OpenFDALoader
from utils.preprocessing import DrugFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
import joblib


def _extract_terms(value):
    text = str(value or "").lower().strip()
    if not text:
        return []

    terms = set()
    for part in re.split(r"[,;/|\n]+", text):
        cleaned = re.sub(r"[^a-z0-9\s-]", " ", part)
        cleaned = " ".join(cleaned.split()).strip()
        if not cleaned:
            continue
        terms.add(cleaned)
        for token in cleaned.split():
            if len(token) >= 4:
                terms.add(token)

    return sorted(terms, key=len, reverse=True)


def _contains_any_term(text, terms):
    haystack = str(text or "").lower()
    for term in terms:
        if len(term) < 4:
            continue
        if re.search(rf"\b{re.escape(term)}\b", haystack):
            return True
    return False


# -------------------------------------------------------
# STEP 1: DOWNLOAD FDA DATA
# -------------------------------------------------------

print("\n[Step 1/6] Downloading FDA Data...\n")

loader = OpenFDALoader()

drug_catalog = {

    # Pain & Inflammation
    "Pain/Inflammation (NSAIDs & Analgesics)": [
        "Aspirin","Ibuprofen","Naproxen","Acetaminophen","Diclofenac",
        "Celecoxib","Meloxicam","Indomethacin","Ketorolac",
        "Etodolac","Piroxicam","Sulindac",
        "Ketoprofen","Oxaprozin","Tolmetin",
        "Meclofenamate","Mefenamic Acid",
        "Tramadol","Tapentadol",
        "Oxycodone","Hydrocodone",
        "Fentanyl","Buprenorphine",
        "Hydromorphone","Methadone"
    ],

    # Antibiotics
    "Antibiotics": [
        "Amoxicillin","Amoxicillin Clavulanate","Ampicillin",
        "Penicillin G","Penicillin V",
        "Azithromycin","Clarithromycin","Erythromycin","Roxithromycin",
        "Ciprofloxacin","Levofloxacin","Moxifloxacin","Ofloxacin",
        "Doxycycline","Minocycline","Tetracycline",
        "Cephalexin","Ceftriaxone","Cefixime","Cefuroxime","Cefepime","Cefpodoxime",
        "Clindamycin","Metronidazole",
        "Trimethoprim","Sulfamethoxazole",
        "Vancomycin","Linezolid","Daptomycin",
        "Meropenem","Imipenem","Ertapenem",
        "Piperacillin","Tazobactam",
        "Nitrofurantoin","Tigecycline"
    ],

    # Cardiovascular
    "Cardiovascular (BP & Heart)": [
        "Lisinopril","Enalapril","Ramipril","Perindopril","Quinapril","Benazepril",
        "Losartan","Valsartan","Olmesartan","Telmisartan","Irbesartan","Candesartan",
        "Amlodipine","Nifedipine","Diltiazem","Verapamil","Felodipine","Nicardipine",
        "Metoprolol","Atenolol","Bisoprolol","Carvedilol","Nebivolol","Propranolol","Nadolol",
        "Hydrochlorothiazide","Chlorthalidone","Indapamide",
        "Furosemide","Torsemide","Bumetanide",
        "Spironolactone","Eplerenone","Amiloride",
        "Digoxin","Ivabradine",
        "Clonidine","Methyldopa",
        "Hydralazine","Minoxidil",
        "Doxazosin","Prazosin","Terazosin"
    ],

    # Diabetes
    "Diabetes": [
        "Metformin","Glipizide","Glyburide","Glimepiride",
        "Insulin","Insulin Glargine","Insulin Lispro","Insulin Aspart",
        "Sitagliptin","Linagliptin","Saxagliptin","Alogliptin",
        "Pioglitazone","Rosiglitazone",
        "Empagliflozin","Dapagliflozin","Canagliflozin",
        "Liraglutide","Semaglutide","Dulaglutide",
        "Exenatide"
    ],

    # Cholesterol
    "Cholesterol (Statins & Lipids)": [
        "Atorvastatin","Simvastatin","Rosuvastatin",
        "Pravastatin","Lovastatin","Fluvastatin","Pitavastatin",
        "Ezetimibe","Fenofibrate","Gemfibrozil",
        "Alirocumab","Evolocumab",
        "Bempedoic Acid"
    ],

    # Gastrointestinal
    "Stomach/GI": [
        "Omeprazole","Pantoprazole","Esomeprazole","Lansoprazole","Rabeprazole",
        "Ranitidine","Famotidine","Cimetidine",
        "Sucralfate","Domperidone","Metoclopramide",
        "Ondansetron","Granisetron",
        "Loperamide","Diphenoxylate",
        "Mesalamine","Sulfasalazine"
    ],

    # Blood Thinners
    "Blood Thinners": [
        "Warfarin","Heparin","Enoxaparin","Dalteparin",
        "Clopidogrel","Prasugrel","Ticagrelor",
        "Aspirin Low Dose",
        "Apixaban","Rivaroxaban","Dabigatran","Edoxaban"
    ],

    # Mental Health
    "Mental Health": [
        "Sertraline","Fluoxetine","Paroxetine","Citalopram","Escitalopram",
        "Venlafaxine","Duloxetine","Desvenlafaxine",
        "Amitriptyline","Imipramine","Nortriptyline",
        "Bupropion","Mirtazapine","Trazodone",
        "Alprazolam","Lorazepam","Diazepam","Clonazepam",
        "Zolpidem","Zopiclone","Buspirone"
    ],

    # Antipsychotics
    "Antipsychotics": [
        "Risperidone","Olanzapine","Quetiapine",
        "Aripiprazole","Clozapine",
        "Haloperidol","Ziprasidone",
        "Paliperidone","Lurasidone","Asenapine"
    ],

    # Antiepileptics
    "Antiepileptics": [
        "Phenytoin","Carbamazepine","Valproate",
        "Lamotrigine","Levetiracetam","Topiramate",
        "Phenobarbital","Oxcarbazepine",
        "Lacosamide","Clobazam","Ethosuximide"
    ],

    # Antifungals
    "Antifungals": [
        "Fluconazole","Ketoconazole","Itraconazole",
        "Voriconazole","Posaconazole","Isavuconazole",
        "Clotrimazole","Miconazole","Nystatin",
        "Amphotericin B","Caspofungin","Micafungin"
    ],

    # Antivirals
    "Antivirals": [
        "Acyclovir","Valacyclovir","Famciclovir",
        "Oseltamivir","Zanamivir",
        "Ritonavir","Lopinavir","Darunavir","Atazanavir",
        "Remdesivir","Sofosbuvir","Ledipasvir"
    ],

    # Respiratory
    "Respiratory": [
    "Albuterol", "Levalbuterol", "Terbutaline",
    "Salmeterol", "Formoterol", "Vilanterol", "Olodaterol", "Indacaterol",
    "Ipratropium",
    "Tiotropium", "Aclidinium", "Umeclidinium", "Glycopyrrolate",
    "Budesonide", "Fluticasone", "Mometasone",
    "Beclomethasone", "Ciclesonide",
    "Fluticasone/Salmeterol",
    "Budesonide/Formoterol",
    "Fluticasone/Vilanterol",
    "Mometasone/Formoterol",
    "Umeclidinium/Vilanterol",
    "Tiotropium/Olodaterol",
    "Montelukast", "Zafirlukast", "Zileuton",
    "Theophylline", "Aminophylline",
    "Omalizumab", "Mepolizumab", "Reslizumab",
    "Benralizumab", "Dupilumab", "Tezepelumab",
    "Roflumilast",
    "Cromolyn",
    "Acetylcysteine", "Guaifenesin",
    "Sildenafil", "Tadalafil", "Bosentan",
    "Ambrisentan", "Riociguat"
],

    # Steroids
    "Steroids": [
    "Prednisone", "Prednisolone",
    "Methylprednisolone", "Dexamethasone",
    "Hydrocortisone", "Betamethasone",
    "Triamcinolone", "Deflazacort",
    "Fludrocortisone",
    "Cortisone", "Paramethasone",
    "Clobetasol", "Halobetasol",
    "Fluocinonide", "Fluocinolone",
    "Desonide", "Desoximetasone",
    "Mometasone", "Fluticasone",
    "Beclomethasone", "Budesonide",
    "Ciclesonide",
    "Medroxyprogesterone",
    "Norethindrone",
    "Estradiol",
    "Testosterone",
    "Danazol",
    "Anastrozole", "Letrozole",
    "Tamoxifen",
    "Abiraterone"
],

    # Thyroid
    "Thyroid": [
    "Levothyroxine",
    "Liothyronine",
    "Methimazole",
    "Propylthiouracil",
    "Liotrix",
    "Thyroid Desiccated",
    "Carbimazole",
    "Potassium Iodide",
    "Iodine",
    "Sodium Iodide I-131",
    "Sodium Iodide I-123",
    "Thyrotropin Alfa"
],

    # Immunosuppressants
    "Immunosuppressants": [
        "Methotrexate","Cyclosporine",
        "Tacrolimus","Azathioprine",
        "Mycophenolate","Sirolimus",
        "Everolimus"
    ],

    # Oncology
    "Oncology (Common)": [
        "Cyclophosphamide","Doxorubicin","Cisplatin",
        "Carboplatin","Paclitaxel","Docetaxel",
        "Methotrexate Oncology","5-Fluorouracil",
        "Capecitabine","Vincristine","Vinblastine",
        "Bleomycin","Etoposide","Imatinib","Dasatinib"
    ],

    # Allergy & Misc
    "Other Common": [
        "Gabapentin","Pregabalin",
        "Tramadol","Codeine","Morphine",
        "Cetirizine","Loratadine","Fexofenadine",
        "Diphenhydramine","Hydroxyzine",
        "Baclofen","Tizanidine",
        "Allopurinol","Colchicine"
    ]
}

# Flatten category->drug mapping to a unique list of drug names.
drug_list = []
for _, names in drug_catalog.items():
    drug_list.extend(names)
drug_list = list(dict.fromkeys(drug_list))

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

# Ensure required columns exist even when API returns sparse payloads.
for col in [
    "brand_name",
    "generic_name",
    "drug_interactions",
    "active_ingredient",
    "pharm_class_epc",
]:
    if col not in df.columns:
        df[col] = ""

df["drug_name"] = (
    df["generic_name"]
    .where(df["generic_name"].astype(str).str.strip() != "", df["brand_name"])
    .astype(str)
    .str.lower()
    .str.strip()
)
df["drug_interactions_text"] = df["drug_interactions"].astype(str)
df = df[df["drug_name"] != ""].drop_duplicates(subset=["drug_name"]).reset_index(drop=True)

df.to_csv("data/processed/drugs_database.csv", index=False)

print("Saved full processed database.")


# -------------------------------------------------------
# STEP 3: GENERATE IMPROVED INTERACTIONS
# -------------------------------------------------------

print("\n[Step 3/6] Generating interaction dataset...\n")

interaction_pairs = []
drug_names = df["drug_name"].tolist()
drug_meta = (
    df.set_index("drug_name")[
        ["drug_interactions_text", "active_ingredient", "pharm_class_epc", "generic_name"]
    ]
    .to_dict("index")
)

for i in range(len(drug_names)):
    for j in range(i + 1, len(drug_names)):

        d1 = drug_names[i]
        d2 = drug_names[j]

        m1 = drug_meta[d1]
        m2 = drug_meta[d2]
        text1 = m1["drug_interactions_text"]

        ingredient1_terms = _extract_terms(m1["active_ingredient"])
        ingredient2_terms = _extract_terms(m2["active_ingredient"])
        generic2_terms = _extract_terms(m2["generic_name"])

        # Positive if same ingredient / same class / ingredient mention in interaction text.
        label = 0
        if ingredient1_terms and ingredient2_terms and set(ingredient1_terms) & set(ingredient2_terms):
            label = 1
        elif (
            str(m1["pharm_class_epc"]).strip()
            and str(m2["pharm_class_epc"]).strip()
            and str(m1["pharm_class_epc"]).strip().lower() == str(m2["pharm_class_epc"]).strip().lower()
        ):
            label = 1
        elif _contains_any_term(text1, ingredient2_terms):
            label = 1
        elif _contains_any_term(text1, generic2_terms):
            label = 1

        interaction_pairs.append({
            "drug1": d1,
            "drug2": d2,
            "label": label
        })

interaction_df = pd.DataFrame(interaction_pairs)

# -------------------------------------------------------
# Balance Dataset
# -------------------------------------------------------

positive = interaction_df[interaction_df["label"] == 1]
negative = interaction_df[interaction_df["label"] == 0]

if len(positive) > 0:
    # Recall-focused balance: keep class ratio closer to 1:1.
    negative = negative.sample(min(len(negative), len(positive)), random_state=42)
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
    raise ValueError("Dataset still not balanced. No positive samples found.")

print(f"Feature matrix size: {len(X)} samples")


# -------------------------------------------------------
# STEP 5: TRAIN MODEL
# -------------------------------------------------------

print("\n[Step 5/6] Training ML model...\n")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=18,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42
)

model.fit(X_train, y_train)

# Tune threshold on validation set to improve balanced classification quality.
val_probs = model.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_f1 = -1.0
for threshold in np.linspace(0.30, 0.75, 46):
    val_pred = (val_probs >= threshold).astype(int)
    score = f1_score(y_val, val_pred, zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

test_probs = model.predict_proba(X_test)[:, 1]
y_pred = (test_probs >= best_threshold).astype(int)

print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Positive class recall: {recall_score(y_test, y_pred, zero_division=0):.3f}")


# -------------------------------------------------------
# STEP 6: SAVE MODEL
# -------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("data/models", exist_ok=True)
legacy_model_path = "models/drug_interaction_model.pkl"
app_model_path = "data/models/drug_classifier.pkl"
joblib.dump(model, legacy_model_path)
joblib.dump(model, app_model_path)

with open("data/models/drug_classifier_meta.json", "w", encoding="utf-8") as f:
    json.dump({"threshold": best_threshold}, f, indent=2)

print(f"\nModel saved to {legacy_model_path}")
print(f"Model saved to {app_model_path}")
print("Model metadata saved to data/models/drug_classifier_meta.json")
print("\n[Step 6/6] Pipeline Complete 🚀")
