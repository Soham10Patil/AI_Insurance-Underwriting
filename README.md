# 🛡️ Insurance Underwriting Decision System

![SHAP Explanation](static/shap_force_plot.png)

A machine learning-powered system to automate insurance underwriting decisions using **Explainable AI (XAI)**. It leverages SHAP values to provide transparent, auditable insights into risk-based decisions for insurance applicants.

---

## 🚀 Features

- **Risk Assessment**: Evaluates applicants based on health, demographic, and financial metrics.
- **Explainable AI**: SHAP-based interpretability to explain decision reasoning.
- **Decision Overrides**: Manual override feature with audit trails for compliance.
- **Visual Dashboard**: UI for underwriters to review, interpret, and manage applications.
- **Audit Logging**: Comprehensive logs for every decision and override action.

---

## 🧱 System Architecture

```
insurance-underwriting/
├── app.py                  # Flask app (API server)
├── train_model.py          # Model training & synthetic data generation
├── underwriting_model.pkl  # Trained ML model
├── scaler.pkl              # Feature scaler for input data
├── shap_explainer.pkl      # SHAP explainer object
├── model_metadata.json     # Metadata and config
├── requirements.txt        # Project dependencies
├── static/                 # Images, CSS, SHAP plots
├── templates/              # HTML templates for dashboard
└── data/                   # Logs of decisions and overrides
```

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/insurance-underwriting.git
   cd insurance-underwriting
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (generates synthetic data and trains the classifier):
   ```bash
   python train_model.py
   ```

4. **Run the Flask server**:
   ```bash
   python app.py
   ```

5. Open your browser and visit: [http://localhost:5000](http://localhost:5000)

---

## 📡 API Endpoints

### 🔍 Underwriting Decision
- `POST /api/underwrite`
- **Input** (JSON):
  ```json
  {
    "applicant_id": "APP12345",
    "age": 42,
    "gender": "M",
    "bmi": 28.5,
    "smoker": 0,
    "has_diabetes": 0,
    "has_heart_disease": 0,
    "blood_pressure": 125,
    "income": 85000,
    "credit_score": 720,
    "debt_to_income": 0.25
  }
  ```
- **Output**:
  ```json
  {
    "applicant_id": "APP12345",
    "decision": "approve",
    "confidence": {
      "approve": 0.82,
      "approve_with_conditions": 0.15,
      "decline": 0.03
    },
    "key_factors": {
      "has_heart_disease": -0.21,
      "credit_score": 0.18,
      "smoker": -0.15,
      "age": 0.12,
      "income": 0.09
    },
    "shap_plot_url": "/static/shap_waterfall_12345.png",
    "timestamp": "2024-03-15T14:30:00Z"
  }
  ```

---

### ✍️ Decision Override
- `POST /api/override`
- Input: JSON with override details
- Output: Status of operation

---

### 📑 Audit Log
- `GET /api/audit` – Returns all past decisions.
- `GET /api/download-csv` – Download decision history as CSV.

---

## 🧠 Model Details

- **Model**: Random Forest Classifier
- **Training Data**: Synthetic data simulating real insurance applicant profiles

### 🔢 Input Features

- **Demographics**: `age`, `gender`
- **Health**: `bmi`, `smoker`, `has_diabetes`, `has_heart_disease`, `blood_pressure`
- **Financial**: `income`, `credit_score`, `debt_to_income`

### 🧾 Decision Classes

- `approve`: Low-risk applicants
- `approve_with_conditions`: Moderate risk (e.g., conditional premiums)
- `decline`: High-risk applicants

---


## 📬 Next Steps

To complete your setup:

1. Place real screenshots into the `/static` folder and update the paths above.
2. Replace the GitHub repository URL with your actual link.
3. Add a proper `LICENSE` file if you haven't already.
4. Optionally include:
   - **Deployment instructions**
   - **Contributing guidelines**
   - **Contact or support info**

---

## 🤝 Contributions

Pull requests and feedback are welcome! 

---
