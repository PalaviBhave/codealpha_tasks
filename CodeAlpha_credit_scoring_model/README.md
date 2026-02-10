# Credit Scoring Model

A machine learning project that predicts credit worthiness using various financial indicators. This project implements multiple ML algorithms including Logistic Regression, Decision Tree, and Random Forest to assess whether a loan applicant is creditworthy based on their financial profile.

## 📊 Project Overview

This credit scoring system uses historical financial data to predict the likelihood of a person being creditworthy. The model analyzes features such as income, debts, payment history, age, and late payment records to make informed predictions.

### Key Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, and Random Forest
- **Feature Engineering**: Custom financial ratios and weighted scores
- **Comprehensive Evaluation**: Precision, Recall, F1-Score, and ROC-AUC metrics
- **Visualization**: Confusion matrix for model performance analysis
- **Model Persistence**: Trained models saved for future predictions

## 📁 Project Structure

```
credit_scoring/
│
├── dataset.py                    # Generate synthetic credit dataset
├── features.py                   # Feature engineering script
├── train.py                      # Train multiple ML models
├── evaluate.py                   # Evaluate model performance
│
├── credit_data.csv              # Raw dataset (1000 samples)
├── credit_data_engineered.csv   # Dataset with engineered features
├── confusion_matrix.png         # Visual model evaluation
│
└── saved/                       # Trained model artifacts
    ├── models.pkl              # All trained models
    ├── scaler.pkl              # StandardScaler for Logistic Regression
    └── test_data.pkl           # Test data for evaluation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit_scoring.git
cd credit_scoring
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Or create a requirements.txt:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Step 1: Generate Dataset (Optional)

If you want to regenerate the dataset:
```bash
python dataset.py
```

This creates `credit_data.csv` with 1000 samples containing:
- `income`: Annual income (₹20,000 - ₹120,000)
- `debts`: Total debts (₹0 - ₹50,000)
- `payment_history`: Payment score (0-100, higher is better)
- `age`: Age (18-70 years)
- `num_late_payments`: Number of late payments (0-20)
- `creditworthy`: Target variable (1 = creditworthy, 0 = not creditworthy)

### Step 2: Feature Engineering

Create enhanced features:
```bash
python features.py
```

This generates `credit_data_engineered.csv` with additional features:
- `debt_to_income_ratio`: Debts divided by income
- `payment_score_weighted`: Payment history weighted by debt ratio

### Step 3: Train Models

Train all models:
```bash
python train.py
```

This trains three models:
1. **Logistic Regression** (with StandardScaler)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

All models are saved in the `saved/` directory.

### Step 4: Evaluate Models

Evaluate model performance:
```bash
python evaluate.py
```

This outputs:
- Performance metrics table for all models
- Detailed classification report for Random Forest
- Confusion matrix visualization saved as `confusion_matrix.png`

## 📈 Model Performance

Expected performance metrics (based on synthetic data):

| Model                | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | ~0.85     | ~0.83  | ~0.84    | ~0.91   |
| Decision Tree       | ~0.82     | ~0.80  | ~0.81    | ~0.87   |
| Random Forest       | ~0.87     | ~0.85  | ~0.86    | ~0.93   |

**Note**: These metrics are based on synthetic data. Real-world performance may vary.

## 🔧 Code Examples

### Making Predictions with a Trained Model

```python
import joblib
import pandas as pd

# Load the trained models and scaler
models = joblib.load("saved/models.pkl")
scaler = joblib.load("saved/scaler.pkl")

# Create a new applicant profile
new_applicant = pd.DataFrame({
    'income': [50000],
    'debts': [10000],
    'payment_history': [85],
    'age': [35],
    'num_late_payments': [2],
    'debt_to_income_ratio': [0.2],
    'payment_score_weighted': [68.0]
})

# Use Random Forest for prediction (best performing model)
rf_model = models["Random Forest"]
prediction = rf_model.predict(new_applicant)
probability = rf_model.predict_proba(new_applicant)

print(f"Creditworthy: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Confidence: {probability[0][1] * 100:.2f}%")
```

### Understanding the Target Variable

The `creditworthy` label is determined by a weighted score:
- **Income** (40% weight): Higher income → better creditworthiness
- **Payment History** (40% weight): Better payment record → better creditworthiness
- **Debts** (20% weight): Lower debts → better creditworthiness

If the composite score > 0.5, the applicant is labeled as creditworthy (1), otherwise not creditworthy (0).

## 🛠️ Technologies Used

- **Python 3.x**: Programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and metrics
- **matplotlib**: Data visualization
- **joblib**: Model serialization

## 📊 Dataset Description

### Features:
- **income**: Annual income in rupees
- **debts**: Total outstanding debts
- **payment_history**: Score from 0-100 indicating payment reliability
- **age**: Applicant's age
- **num_late_payments**: Count of late payments
- **debt_to_income_ratio**: Engineered feature (debts/income)
- **payment_score_weighted**: Engineered feature (payment_history × (1 - debt_to_income_ratio))

### Target:
- **creditworthy**: Binary classification (1 = Yes, 0 = No)

## 🔍 Model Details

### Logistic Regression
- Uses StandardScaler for feature normalization
- Linear decision boundary
- Provides probability estimates
- Fast training and prediction

### Decision Tree
- No scaling required
- Interpretable decision rules
- Can capture non-linear relationships
- Prone to overfitting without pruning

### Random Forest
- Ensemble of decision trees
- Reduces overfitting through averaging
- Handles non-linear relationships well
- Generally the best performing model

## 📝 Future Enhancements

- [ ] Add more features (credit utilization, employment status, etc.)
- [ ] Implement cross-validation for robust evaluation
- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Create a web API using Flask/FastAPI
- [ ] Add SHAP values for model explainability
- [ ] Implement additional algorithms (XGBoost, LightGBM)
- [ ] Add real-world dataset integration
- [ ] Create interactive dashboard (Streamlit/Dash)




## 👤 Author

Your Name - @PalaviBhave

Project Link: [https://github.com/PalaviBhave/credit_scoring](https://github.com/PalaviBhave/credit_scoring)




