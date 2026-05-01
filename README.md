# 🧬 AI/ML Drug Interaction Checker

An advanced AI/ML-powered application for predicting drug interactions using Machine Learning and Deep Learning models.

## 🎯 Features

- **Machine Learning Models**
  - Random Forest Classifier
  - Neural Network (PyTorch)
  - Feature Engineering
  - Model Training & Evaluation

- **Smart Predictions**
  - ML-based interaction prediction
  - Confidence scoring
  - Risk level assessment
  - Fallback to rule-based system

- **User Interface**
  - Interactive Streamlit web app
  - Real-time predictions
  - Detailed drug information
  - Report generation

---
    
## 🛠️ Tech Stack  

### 💻 Programming  
- Python  

### 📊 Libraries & Tools  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  

### ⚙️ Environment  
- Jupyter Notebook / VS Code

---

## ⚙️ Workflow  

1. Data Collection  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Prediction of Drug Interactions

---

## 📁 Project Structure

```
drug-interaction-aiml/
│
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   ├── processed/                 # Processed datasets
│   └── models/                    # Trained models
│       ├── drug_classifier.pkl    # Random Forest model
│       └── interaction_predictor.pth  # Neural Network model
│
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── drug_classifier.py         # Random Forest classifier
│   ├── interaction_predictor.py   # Neural Network
│   └── text_analyzer.py           # NLP components
│
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Feature engineering
│   └── database.py                # Drug information database
│
├── app/                           # Application code
│   ├── __init__.py
│   └── streamlit_app.py           # Main Streamlit app
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                         # Unit tests
│   ├── test_models.py
│   └── test_utils.py
│
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── train.py                       # Training script
├── predict.py                     # Prediction script
└── README.md                      # This file
```

## 🚀 Installation

### 1. Create Virtual Environment

```bash
# Using conda
conda create -n drug-aiml python=3.10 -y
conda activate drug-aiml

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Required Directories

```bash
mkdir -p data/raw data/processed data/models logs
```

## 📊 Usage

### Training Models

Train both Random Forest and Neural Network models:

```bash
python train.py
```

This will:
- Load drug interaction data
- Build training dataset
- Train Random Forest classifier
- Train Neural Network
- Evaluate models
- Save trained models to `data/models/`

### Running the Application

Start the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

Or if running from root:

```bash
cd app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Making Predictions

Use the prediction script for command-line predictions:

```bash
python predict.py --drug1 "Aspirin" --drug2 "Ibuprofen"
```

## 🧪 Model Architecture

### Random Forest Classifier

- **Algorithm**: Random Forest
- **Features**: 
  - Text features (drug names)
  - Drug properties (category, ingredients)
  - Interaction patterns
- **Output**: Binary classification + probability scores

### Neural Network

- **Architecture**: 
  - Input layer: Feature dimension
  - Hidden layers: [256, 128, 64]
  - Dropout: 0.3
  - Output: Sigmoid activation
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam

## 📈 Feature Engineering

The system extracts various features:

1. **Text Features**
   - Drug name length
   - Character patterns
   - Dosage indicators

2. **Drug Properties**
   - Active ingredients
   - Drug category
   - Therapeutic use
   - Side effect patterns

3. **Interaction Features**
   - Same active ingredient
   - Same drug category
   - Known interaction patterns

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  classifier:
    type: "random_forest"
    n_estimators: 100
    max_depth: 20
    
  neural_network:
    hidden_layers: [256, 128, 64]
    dropout: 0.3
    learning_rate: 0.001
    epochs: 50
    batch_size: 32

data:
  train_test_split: 0.2
  validation_split: 0.1
  random_seed: 42

thresholds:
  high_risk: 0.8
  medium_risk: 0.5
  low_risk: 0.2
```

## 📚 Adding New Drugs

### Via Code

Edit `utils/database.py` and add to the `DRUG_DATABASE`:

```python
"new_drug": {
    "generic": "Active Ingredient",
    "category": "Drug Category",
    "uses": "Medical uses",
    "side_effects": "Side effects",
    "warnings": "Important warnings",
    "active_ingredient": "ingredient_name",
    "dosage_form": "tablet",
    "manufacturer": "Company Name"
}
```

### Via CSV Import

```python
from utils.database import DrugDatabase

db = DrugDatabase()
db.import_from_csv('path/to/drugs.csv')
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# With coverage
pytest --cov=models --cov=utils tests/
```

## 📊 Model Performance

Current models achieve:

- **Random Forest**
  - Accuracy: ~85-90%
  - Precision: ~88%
  - Recall: ~85%
  - AUC-ROC: ~0.90

- **Neural Network**
  - Accuracy: ~87-92%
  - Precision: ~90%
  - Recall: ~87%
  - AUC-ROC: ~0.92

*Note: Performance depends on dataset quality and size*

## 🔬 Data Sources

The system uses:
- Built-in drug database (expandable)
- Custom interaction data
- Can integrate with external APIs:
  - FDA Drug Database
  - DrugBank
  - RxNorm

## ⚠️ Important Disclaimers

1. **Medical Advice**: This tool is for educational purposes only
2. **Not a Substitute**: Always consult healthcare professionals
3. **Accuracy**: ML models may not catch all interactions
4. **Emergency**: Call local emergency services for urgent medical needs

## 🔒 Safety Features

- Multiple validation layers
- Duplicate ingredient detection
- Rule-based fallback system
- Confidence scoring
- Risk level assessment

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 🎓 Educational Use

Perfect for:
- ML/AI learning projects
- Healthcare technology courses
- Feature engineering practice
- PyTorch/Scikit-learn tutorials

## 🔄 Future Enhancements

- [ ] Integration with real medical databases
- [ ] More sophisticated NLP models
- [ ] User feedback mechanism
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] API endpoint for developers

## 📚 References

- Scikit-learn Documentation
- PyTorch Tutorials
- Streamlit Documentation
- Drug Interaction Databases  

---

## 📊 Model Details  
- Algorithm Used: (e.g., Random Forest / Neural Network )  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score  

---

