"""
AI/ML-Powered Drug Interaction Checker
Main Streamlit Application
"""

import streamlit as st
import sys
import os
import json
import numpy as np
import pandas as pd
import yaml
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be first Streamlit command
# Must be first Streamlit command
st.set_page_config(
    page_title="AI Drug Interaction Checker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# UI STYLING (STEP 1)
# ================================
def load_ui_css():
    bg = "#0F172A"
    card = "rgba(15, 23, 42, 0.55)"
    text = "#E6F7FF"
    muted = "#A8C5D1"
    accent = "#00F5D4"
    success = "#22C55E"
    warning = "#FACC15"
    input_bg = "rgba(15, 23, 42, 0.72)"
    hero_bg = "linear-gradient(135deg, rgba(0,245,212,0.14), rgba(15,23,42,0.85))"
    hero_border = "rgba(0,245,212,0.35)"
    hero_title = "#E6F7FF"
    hero_text = "#B8D7E5"
    metric_border = "rgba(0,245,212,0.28)"

    st.markdown("""
    <style>
    :root {
        --bg: BG_COLOR;
        --card: CARD_COLOR;
        --text: TEXT_COLOR;
        --muted: MUTED_COLOR;
        --accent: ACCENT_COLOR;
        --success: SUCCESS_COLOR;
        --warning: WARNING_COLOR;
        --input-bg: INPUT_BG;
        --metric-border: METRIC_BORDER;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        position: relative;
        overflow-x: hidden;
        background:
          radial-gradient(circle at 15% 10%, rgba(0,245,212,0.12) 0%, transparent 35%),
          radial-gradient(circle at 85% 15%, rgba(34,197,94,0.12) 0%, transparent 35%),
          var(--bg);
    }
    .stApp::before, [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: -20%;
        background: linear-gradient(120deg, rgba(0,245,212,0.08), rgba(34,197,94,0.05), rgba(15,23,42,0.0));
        filter: blur(60px);
        animation: auroraShift 12s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes auroraShift {
        0% { transform: translateX(-5%) translateY(-2%); }
        100% { transform: translateX(5%) translateY(2%); }
    }

    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    div[data-testid="stDivider"] {display: none !important;}

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: Inter, "Segoe UI", sans-serif;
    }
    h1, h2, h3 {
        color: var(--text);
        font-weight: 700;
    }
    .block-container, .stMarkdown p, .stCaption, label, small {
        color: var(--text);
    }

    .block-container {
        padding-top: 1.2rem;
        position: relative;
        z-index: 1;
    }
    [data-testid="stMain"], [data-testid="stAppViewBlockContainer"], .main {
        background: transparent !important;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--metric-border);
        border-radius: 14px;
        padding: 0.6rem 0.9rem;
        backdrop-filter: blur(14px);
        box-shadow: 0 0 0 1px rgba(0,245,212,0.08), 0 10px 30px rgba(0,0,0,0.35);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    .stAlert {
        background: var(--card);
        border-radius: 12px;
        border: 1px solid var(--metric-border);
        backdrop-filter: blur(14px);
    }
    .stAlert p, .stAlert span {
        color: var(--text) !important;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(0,245,212,0.35);
        background: linear-gradient(90deg, rgba(0,245,212,0.32), rgba(0,245,212,0.18));
        color: white;
        font-weight: 700;
        letter-spacing: 0.2px;
        box-shadow: 0 0 18px rgba(0,245,212,0.25);
    }

    .stButton > button:hover {
        border-color: rgba(0,245,212,0.7);
        box-shadow: 0 0 26px rgba(0,245,212,0.42);
    }
    .stTextArea textarea, .stTextInput input {
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--metric-border) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: var(--muted) !important;
    }
    div[role="radiogroup"] label, div[role="radiogroup"] span {
        color: var(--text) !important;
    }
    div[data-testid="stInfo"] {
        background: var(--card) !important;
        border: 1px solid var(--metric-border) !important;
        backdrop-filter: blur(12px);
    }
    div[data-testid="stInfo"] p, div[data-testid="stInfo"] li, div[data-testid="stInfo"] span {
        color: var(--text) !important;
    }

    .hero {
        background: HERO_BG;
        border: 1px solid HERO_BORDER;
        border-radius: 18px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 0 30px rgba(0,245,212,0.12);
    }

    .hero h2 {
        margin: 0 0 0.4rem 0;
        color: HERO_TITLE;
        font-family: Orbitron, Inter, "Segoe UI", sans-serif;
        letter-spacing: 0.4px;
    }

    .hero p {
        margin: 0;
        color: HERO_TEXT;
    }
    .status-card {
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        background: var(--card);
        border: 1px solid var(--metric-border);
        backdrop-filter: blur(14px);
        min-height: 94px;
    }
    .status-card .label {
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 0.2rem;
    }
    .status-card .value {
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.1;
    }
    .status-card.active {
        border-color: rgba(34,197,94,0.75);
        box-shadow: 0 0 28px rgba(34,197,94,0.35), inset 0 0 16px rgba(34,197,94,0.12);
    }
    .status-card.active .value {
        color: #86efac;
        text-shadow: 0 0 12px rgba(34,197,94,0.55);
    }
    .section-subhead {
        margin-top: 0.7rem;
        margin-bottom: 0.15rem;
        color: var(--warning);
        font-size: 1.32rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        text-shadow: 0 0 10px rgba(250,204,21,0.22);
    }
    .analysis-subhead {
        margin-top: 0.2rem;
        margin-bottom: 0.35rem;
        color: #9afcf0;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.25px;
        text-transform: uppercase;
    }
    .interaction-title {
        font-size: 2.28rem;
        font-weight: 800;
        color: #e9fbff;
        letter-spacing: 0.2px;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 14px rgba(0,245,212,0.22);
    }
    .interaction-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.35rem 0 0.9rem 0;
    }
    .meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.38rem 0.62rem;
        border-radius: 999px;
        border: 1px solid rgba(0,245,212,0.35);
        background: rgba(15,23,42,0.46);
        color: #e8fbff;
        font-size: 1.12rem;
        font-weight: 700;
    }
    .meta-chip b {
        color: #9afcf0;
        font-weight: 800;
    }
    .chip-high {
        border-color: rgba(239,68,68,0.75);
        box-shadow: 0 0 14px rgba(239,68,68,0.35);
        color: #fecaca;
    }
    .chip-medium {
        border-color: rgba(250,204,21,0.75);
        box-shadow: 0 0 14px rgba(250,204,21,0.30);
        color: #fef08a;
    }
    .chip-low {
        border-color: rgba(34,197,94,0.75);
        box-shadow: 0 0 14px rgba(34,197,94,0.30);
        color: #bbf7d0;
    }
    .chip-confidence { border-color: rgba(0,245,212,0.6); }
    .chip-method { border-color: rgba(34,197,94,0.5); }
    .analysis-body {
        font-size: 1.24rem;
        line-height: 1.72;
        color: #d7edf5;
        margin-bottom: 0.35rem;
    }
    .analysis-bullet {
        font-size: 1.14rem;
        line-height: 1.7;
        color: #d7edf5;
        margin: 0.28rem 0 0.28rem 0.15rem;
    }
    </style>
    """.replace("BG_COLOR", bg)
       .replace("CARD_COLOR", card)
       .replace("TEXT_COLOR", text)
       .replace("MUTED_COLOR", muted)
       .replace("ACCENT_COLOR", accent)
       .replace("SUCCESS_COLOR", success)
       .replace("WARNING_COLOR", warning)
       .replace("INPUT_BG", input_bg)
       .replace("HERO_BG", hero_bg)
       .replace("HERO_BORDER", hero_border)
       .replace("HERO_TITLE", hero_title)
       .replace("HERO_TEXT", hero_text)
       .replace("METRIC_BORDER", metric_border), unsafe_allow_html=True)

# Import custom modules
try:
    from models.drug_classifier import DrugInteractionClassifier
    from utils.database import DrugDatabase
    from utils.preprocessing import DrugFeatureExtractor
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all modules are in the correct directory.")
    st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_risk_level(probability):
    if probability >= 0.8:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    if probability >= 0.2:
        return "LOW"
    return "MINIMAL"


# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from project root config.yaml"""

    try:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        config_path = os.path.join(base_dir, "config.yaml")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    except FileNotFoundError:
        st.warning("config.yaml not found. Using default configuration.")
        return {
            "model": {
                "classifier": {
                    "type": "random_forest"
                }
            },
            "thresholds": {
                "high_risk": 0.8,
                "medium_risk": 0.5,
                "low_risk": 0.2
            }
        }

# Initialize components
@st.cache_resource
def initialize_app():
    """Initialize ML models and database"""
    config = load_config()
    decision_threshold = 0.5

    # Initialize database
    db = DrugDatabase()

    # Initialize ML classifier
    try:
        classifier = DrugInteractionClassifier(
            model_type=config['model']['classifier']['type']
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'data', 'models', 'drug_classifier.pkl')
        legacy_model_path = os.path.join(base_dir, 'models', 'drug_interaction_model.pkl')
        meta_path = os.path.join(base_dir, 'data', 'models', 'drug_classifier_meta.json')

        if os.path.exists(model_path):
            classifier.load_model(model_path)
            logger.info(f"Loaded pre-trained classifier from {model_path}")
        elif os.path.exists(legacy_model_path):
            classifier.load_model(legacy_model_path)
            logger.info(f"Loaded pre-trained classifier from {legacy_model_path}")
        else:
            logger.error(f"Model file not found at {model_path}")

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            decision_threshold = float(meta.get('threshold', 0.5))
            logger.info(f"Using decision threshold: {decision_threshold:.2f}")

    except Exception as e:
        logger.error(f"Error initializing classifier: {e}")
        classifier = None

    # Initialize feature extractor
    feature_extractor = DrugFeatureExtractor()

    return config, db, classifier, feature_extractor, decision_threshold


def predict_interaction_ml(drug1, drug2, classifier, feature_extractor, db, decision_threshold=0.5):
    """
    Predict interaction using:
    1) Clinical rule-based override (FDA / known interactions)
    2) ML model fallback
    """

    # --------------------------------------------------
    # 1️⃣ CLINICAL RULE OVERRIDE (HIGHEST PRIORITY)
    # --------------------------------------------------
    rule_interaction = db.check_interaction(drug1, drug2)

    if rule_interaction:
        if isinstance(rule_interaction, dict):
            confidence = float(rule_interaction.get("confidence", 0.95))
            risk_level = rule_interaction.get("severity", "HIGH")
            description = rule_interaction.get(
                "description",
                "Known clinically significant interaction"
            )
            recommendation = rule_interaction.get(
                "recommendation",
                "Consult healthcare provider"
            )
            evidence_type = rule_interaction.get("evidence_type")
            mechanism_points = rule_interaction.get("mechanism_points", [])
            clinical_risk_points = rule_interaction.get("clinical_risk_points", [])
            monitoring_points = rule_interaction.get("monitoring_points", [])
            management_points = rule_interaction.get("management_points", [])
            research_notes = rule_interaction.get("research_notes", [])
        else:
            confidence = 0.95
            risk_level = "HIGH"
            description = "Known clinically significant interaction"
            recommendation = "Consult healthcare provider"
            evidence_type = None
            mechanism_points = []
            clinical_risk_points = []
            monitoring_points = []
            management_points = []
            research_notes = []

        return {
            "method": "Clinical Rule (FDA / Known Interaction)",
            "has_interaction": True,
            "confidence": confidence,
            "risk_level": risk_level,
            "description": description,
            "recommendation": recommendation,
            "evidence_type": evidence_type,
            "mechanism_points": mechanism_points,
            "clinical_risk_points": clinical_risk_points,
            "monitoring_points": monitoring_points,
            "management_points": management_points,
            "research_notes": research_notes,
        }

    # --------------------------------------------------
    # 2️⃣ DUPLICATE ACTIVE INGREDIENT CHECK
    # --------------------------------------------------
    if db.has_duplicate_ingredient(drug1, drug2):
        return {
            "method": "Ingredient Analysis",
            "has_interaction": True,
            "confidence": 0.99,
            "risk_level": "HIGH",
            "description": "Both medications contain the same active ingredient",
            "recommendation": "Do not take together",
        }

    # --------------------------------------------------
    # 3️⃣ ML FALLBACK (ONLY IF MODEL IS LOADED)
    # --------------------------------------------------
    if classifier is None or classifier.model is None:
        return {
            "method": "Rule-Based Fallback",
            "has_interaction": False,
            "confidence": 0.70,
            "risk_level": "MINIMAL",
            "description": "No known interaction found",
        }

    try:
        features = feature_extractor.extract_pair_features(drug1, drug2, db)

        features_2d = features.reshape(1, -1)
        preds = classifier.predict(features_2d)
        probs = classifier.predict_proba(features_2d)
        prob = probs[0][1] if probs.shape[1] > 1 else probs[0][0]

        return {
            "method": "ML Model",
            "has_interaction": bool(prob >= decision_threshold),
            "confidence": float(prob),
            "risk_level": _get_risk_level(prob),
            "description": f"Predicted using machine learning model (threshold={decision_threshold:.2f})",
        }

    except Exception as e:
        logger.error(f"ML prediction failed: {e}")

        return {
            "method": "Safe Fallback",
            "has_interaction": False,
            "confidence": 0.60,
            "risk_level": "MINIMAL",
            "description": "Prediction failed, using safe default",
        }


def predict_interaction_rules(drug1, drug2, db):
    """
    Fallback rule-based prediction
    
    Args:
        drug1, drug2: Drug names
        db: Drug database
        
    Returns:
        Prediction results dictionary
    """
    # Check for known interactions in database
    interaction = db.check_interaction(drug1, drug2)
    
    if interaction:
        return {
            'method': 'Database Lookup',
            'has_interaction': True,
            'confidence': 0.95,
            'risk_level': 'HIGH',
            'description': interaction
        }
    
    # Check for duplicate ingredients
    if db.has_duplicate_ingredient(drug1, drug2):
        return {
            'method': 'Ingredient Analysis',
            'has_interaction': True,
            'confidence': 0.99,
            'risk_level': 'HIGH',
            'description': 'Both medications contain the same active ingredient'
        }
    
    return {
        'method': 'No Known Interaction',
        'has_interaction': False,
        'confidence': 0.7,
        'risk_level': 'MINIMAL',
        'description': 'No known interactions found'
    }


def display_drug_info(drug_name, db):
    """Display detailed drug information"""
    info = db.get_drug_info(drug_name)

    if info:
        def compact_text(value, max_chars=220):
            text = str(value or "").strip()
            if not text:
                return "N/A"
            text = " ".join(text.split())
            if len(text) <= max_chars:
                return text
            cut = text[:max_chars].rsplit(" ", 1)[0]
            return f"{cut}..."

        def show_value(value):
            text = str(value or "").strip()
            return text if text else "N/A"

        with st.expander(f"Drug Info: {drug_name.title()} - Detailed Information"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Generic Name:** {show_value(info.get('generic'))}")
                st.markdown(f"**Uses (Short):** {compact_text(info.get('uses'))}")
                st.markdown(f"**Warnings (Short):** {compact_text(info.get('warnings'))}")

            with col2:
                st.markdown(f"**Side Effects (Short):** {compact_text(info.get('side_effects'))}")
                st.markdown(f"**Category:** {show_value(info.get('category'))}")
                st.markdown(f"**Contraindications (Short):** {compact_text(info.get('contraindications'))}")


def main():
    """Main application"""
    load_ui_css()

    # Initialize
    config, db, classifier, feature_extractor, decision_threshold = initialize_app()

    st.markdown("""
    <div class="hero">
      <h2>Drug Interaction Analysis Workspace</h2>
      <p>Research-oriented interaction screening with rule-based clinical context and ML-assisted probability scoring.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display model info
    col1, col2, col3 = st.columns(3)
    with col1:
        model_active = bool(classifier and classifier.model)
        if model_active:
            st.markdown(
                """
                <div class="status-card active">
                    <div class="label">Model Status</div>
                    <div class="value">ACTIVE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-card">
                    <div class="label">Model Status</div>
                    <div class="value">RULE-BASED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col2:
        st.metric("Database Entries", db.get_drug_count())
    with col3:
        prediction_method = "Machine Learning" if classifier and classifier.model else "Rule-Based"
        st.metric("Prediction Method", prediction_method)

    # Warning
    st.warning("""
    ⚠️ **MEDICAL DISCLAIMER**: This AI/ML tool is for educational purposes only. 
    Always consult healthcare professionals before making medication decisions.
    """)
    
    st.divider()
    
    # Main interface
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🔬 Analyze Drug Interactions")
        
        # Input methods
        input_method = st.radio(
            "Select input method:",
            ["Manual Entry", "Quick Select"],
            horizontal=True
        )
        
        if input_method == "Manual Entry":
            medications_input = st.text_area(
                "Enter medications (comma-separated):",
                placeholder="e.g., Dolo 650, Aspirin, Crocin",
                height=100
            )
            medications = [m.strip() for m in medications_input.split(',') if m.strip()]
        else:
            # Quick select from database
            available_drugs = db.get_all_drug_names()
            medications = st.multiselect(
                "Select medications:",
                options=available_drugs,
                max_selections=5
            )
        
        analyze_button = st.button("🔍 Analyze Interactions", type="primary", use_container_width=True)
    
    with col_right:
        st.subheader("📋 Quick Info")
        st.info(f"""
        **Available Drugs:** {db.get_drug_count()}
        
        **Analysis Features:**
        - ML-based prediction
        - Risk level assessment
        - Confidence scoring
        - Detailed drug info
        """)
    
    # Analysis
    if analyze_button and medications:
        if len(medications) < 2:
            st.error("Please enter at least 2 medications to check interactions.")
        else:
            st.success(f"Analyzing {len(medications)} medications...")
            
            # Display individual drug info
            st.subheader("💊 Medication Information")
            for drug in medications:
                display_drug_info(drug, db)
            
            st.divider()
            
            # Check interactions
            st.subheader("🔄 Interaction Analysis")
            
            interactions_found = []
            
            # Check all pairs
            for i, drug1 in enumerate(medications):
                for drug2 in medications[i+1:]:
                    result = predict_interaction_ml(
                        drug1, drug2, classifier, feature_extractor, db, decision_threshold
                    )
                    
                    if result['has_interaction']:
                        interactions_found.append({
                            'pair': (drug1, drug2),
                            'result': result
                        })
            
            # Display results
            if interactions_found:
                st.error(f"⚠️ Found {len(interactions_found)} potential interaction(s)")
                
                for interaction in interactions_found:
                    drug1, drug2 = interaction['pair']
                    result = interaction['result']
                    
                    # Color code by risk level
                    risk_colors = {
                        'HIGH': '🔴',
                        'MEDIUM': '🟡',
                        'LOW': '🟢',
                        'MINIMAL': '⚪'
                    }
                    
                    risk_icon = risk_colors.get(result['risk_level'], '⚪')
                    
                    with st.container():
                        risk_class_map = {
                            'HIGH': 'chip-high',
                            'MEDIUM': 'chip-medium',
                            'LOW': 'chip-low',
                        }
                        risk_class = risk_class_map.get(result['risk_level'], '')
                        st.markdown(f"""
                        <div class="analysis-subhead">Interaction Summary</div>
                        <div class="interaction-title">{risk_icon} {drug1} ? {drug2}</div>
                        <div class="interaction-meta">
                          <div class="meta-chip {risk_class}"><b>Risk:</b> {result['risk_level']}</div>
                          <div class="meta-chip chip-confidence"><b>Confidence:</b> {result['confidence']:.1%}</div>
                          <div class="meta-chip chip-method"><b>Method:</b> {result['method']}</div>
                        </div>
                        <div class="analysis-subhead">Clinical Interpretation</div>
                        <div class="analysis-body">{result.get('description', 'Potential interaction detected.')}</div>
                        """, unsafe_allow_html=True)

                        st.markdown('<div class="analysis-subhead">Evidence & Action</div>', unsafe_allow_html=True)
                        if result.get('evidence_type'):
                            st.markdown('<div class="section-subhead">Evidence Type</div>', unsafe_allow_html=True)
                            st.markdown(result['evidence_type'])
                        if result.get('recommendation'):
                            st.markdown('<div class="section-subhead">Recommendation</div>', unsafe_allow_html=True)
                            st.markdown(result['recommendation'])
                        if result.get('mechanism_points'):
                            st.markdown('<div class="section-subhead">Mechanism Points</div>', unsafe_allow_html=True)
                            for item in result['mechanism_points']:
                                st.markdown(f"<div class='analysis-bullet'>• {item}</div>", unsafe_allow_html=True)
                        if result.get('clinical_risk_points'):
                            st.markdown('<div class="section-subhead">Clinical Risk Points</div>', unsafe_allow_html=True)
                            for item in result['clinical_risk_points']:
                                st.markdown(f"<div class='analysis-bullet'>• {item}</div>", unsafe_allow_html=True)
                        if result.get('monitoring_points'):
                            st.markdown('<div class="section-subhead">Monitoring Points</div>', unsafe_allow_html=True)
                            for item in result['monitoring_points']:
                                st.markdown(f"<div class='analysis-bullet'>• {item}</div>", unsafe_allow_html=True)
                        if result.get('management_points'):
                            st.markdown('<div class="section-subhead">Management Points</div>', unsafe_allow_html=True)
                            for item in result['management_points']:
                                st.markdown(f"<div class='analysis-bullet'>• {item}</div>", unsafe_allow_html=True)
                        if result.get('research_notes'):
                            st.markdown('<div class="section-subhead">Research Notes</div>', unsafe_allow_html=True)
                            for item in result['research_notes']:
                                st.markdown(f"<div class='analysis-bullet'>• {item}</div>", unsafe_allow_html=True)

                        st.divider()
            else:
                st.success("✅ No significant interactions detected between the selected medications.")
            
            # Download report
            st.divider()
            
            report_content = f"""AI/ML DRUG INTERACTION REPORT
{'='*60}

Medications Analyzed: {', '.join(medications)}
Number of Interactions: {len(interactions_found)}
Prediction Method: {prediction_method}

DETAILED ANALYSIS:
"""
            
            for interaction in interactions_found:
                drug1, drug2 = interaction['pair']
                result = interaction['result']
                report_content += f"""
{drug1} ↔ {drug2}
Risk Level: {result['risk_level']}
Confidence: {result['confidence']:.1%}
Method: {result['method']}
Description: {result.get('description', 'N/A')}
Recommendation: {result.get('recommendation', 'N/A')}
Mechanism Points: {' | '.join(result.get('mechanism_points', [])) if result.get('mechanism_points') else 'N/A'}
Clinical Risk Points: {' | '.join(result.get('clinical_risk_points', [])) if result.get('clinical_risk_points') else 'N/A'}
Monitoring Points: {' | '.join(result.get('monitoring_points', [])) if result.get('monitoring_points') else 'N/A'}
Management Points: {' | '.join(result.get('management_points', [])) if result.get('management_points') else 'N/A'}
Research Notes: {' | '.join(result.get('research_notes', [])) if result.get('research_notes') else 'N/A'}
---
"""
            
            report_content += """
DISCLAIMER:
This AI/ML analysis is for educational purposes only.
Always consult healthcare professionals for medical advice.
"""
            
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name="drug_interaction_report_ml.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        
        st.markdown("""
        This application uses **Machine Learning** to predict drug interactions.
        
        **Features:**
        - 🤖 ML-based predictions
        - 📊 Confidence scoring
        - 🎯 Risk level assessment
        - 📚 Comprehensive database
        
        **Technology Stack:**
        - Random Forest Classifier
        - Neural Networks (PyTorch)
        - Feature Engineering
        - NLP for text analysis
        """)
        
        st.divider()
        
        st.header("📈 Model Performance")
        if classifier and classifier.model:
            st.success("ML Model: Active")
            
            # Show feature importance if available
            importance = classifier.get_feature_importance()
            if importance is not None:
                st.caption("Top features used in prediction")
        else:
            st.warning("Using rule-based fallback")
        
        st.divider()
        
        st.header("🔗 Resources")
        st.markdown("""
        - [Project GitHub](#)
        - [Model Documentation](#)
        - [API Documentation](#)
        - [Drugs.com](https://www.drugs.com)
        """)


if __name__ == "__main__":
    main()
