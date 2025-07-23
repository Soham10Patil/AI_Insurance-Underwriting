import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import shap
import matplotlib.pyplot as plt

# 1. Generate synthetic insurance application data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    data = {
        # Demographics
        'age': np.random.normal(45, 15, num_samples).astype(int),
        'gender': np.random.choice(['M', 'F'], num_samples),
        'bmi': np.random.normal(26, 5, num_samples),
        
        # Health factors
        'smoker': np.random.choice([0, 1], num_samples, p=[0.75, 0.25]),
        'has_diabetes': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'has_heart_disease': np.random.choice([0, 1], num_samples, p=[0.93, 0.07]),
        'blood_pressure': np.random.normal(120, 15, num_samples).astype(int),
        
        # Financial factors
        'income': np.random.normal(70000, 30000, num_samples).astype(int),
        'credit_score': np.random.normal(700, 100, num_samples).astype(int),
        'debt_to_income': np.random.normal(0.3, 0.15, num_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Apply reasonable constraints
    df['age'] = df['age'].clip(18, 85)
    df['bmi'] = df['bmi'].clip(15, 50)
    df['blood_pressure'] = df['blood_pressure'].clip(80, 200)
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 1)
    
    # Generate target variable (approval decision) based on risk factors
    def calculate_decision(row):
        risk_score = 0
        
        # Age factor
        if row['age'] < 25:
            risk_score += 10
        elif row['age'] > 65:
            risk_score += 15
        
        # Health factors
        if row['smoker'] == 1:
            risk_score += 20
        if row['has_diabetes'] == 1:
            risk_score += 15
        if row['has_heart_disease'] == 1:
            risk_score += 25
        if row['bmi'] > 35:
            risk_score += 10
        if row['blood_pressure'] > 140:
            risk_score += 10
        
        # Financial factors
        if row['credit_score'] < 600:
            risk_score += 15
        if row['debt_to_income'] > 0.5:
            risk_score += 10
        
        # Decision logic
        if risk_score < 20:
            return 'approve'  # Low risk - automatic approval
        elif risk_score < 40:
            return 'approve_with_conditions'  # Medium risk - conditional approval
        else:
            return 'decline'  # High risk - declined
    
    df['decision'] = df.apply(calculate_decision, axis=1)
    
    return df

# 2. Build and train a basic model
def train_underwriting_model(data):
    # Prepare features and target
    X = data.drop(['decision'], axis=1)
    y = data['decision']
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'blood_pressure', 'income', 'credit_score', 'debt_to_income']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample of the test set
    sample_size = min(100, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, class_names=model.classes_, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Plot decision plots for a few examples
    # for i in range(3):
    #     plt.figure(figsize=(12, 6))
    #     true_class = y_test.iloc[i]
    #     true_class_index = list(model.classes_).index(true_class)
    #     shap.decision_plot(explainer.expected_value[true_class_index],
    #                      shap_values[true_class_index][i], # Index by class, then sample
    #                      X_sample.iloc[i], feature_names=X_sample.columns, show=False)
    #     plt.tight_layout()
    #     plt.savefig(f'shap_decision_plot_{i}.png')
    #     plt.close()
    
    # Save model, scaler, and explainer
    pickle.dump(model, open('underwriting_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(explainer, open('shap_explainer.pkl', 'wb'))
    
    # Save model metadata (column names and order)
    with open('model_metadata.json', 'w') as f:
        json.dump({
            'feature_names': X_train.columns.tolist(),
            'target_classes': model.classes_.tolist()
        }, f)
    
    return model, scaler, explainer

# 3. Enhanced inference function with SHAP values
def predict_application(application_data, model, scaler, explainer):
    # Prepare the input data
    input_df = pd.DataFrame([application_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Load column names expected by the model
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Ensure input has all expected columns
    for col in metadata['feature_names']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Keep only the columns the model expects and in the right order
    input_df = input_df[metadata['feature_names']]
    
    # Scale numerical features
    numerical_cols = ['age', 'bmi', 'blood_pressure', 'income', 'credit_score', 'debt_to_income']
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = scaler.transform(input_df[[col]])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Map prediction probabilities to class names
    confidence_scores = {cls: float(prob) for cls, prob in zip(model.classes_, prediction_proba)}
    
    # Calculate SHAP values for this prediction
    shap_values = explainer.shap_values(input_df)
    
    # Get the index of the predicted class
    pred_class_idx = list(model.classes_).index(prediction)
    
    # Get SHAP values for the predicted class
    pred_shap_values = shap_values[pred_class_idx][0]
    
    # Create a dictionary of feature names and their SHAP values
    shap_explanation = {feature: float(shap_val) for feature, shap_val in zip(metadata['feature_names'], pred_shap_values)}
    
    # Sort by absolute SHAP value to find most important features
    key_factors = dict(sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    
    return {
        'decision': prediction,
        'confidence': confidence_scores,
        'key_factors': key_factors,
        'shap_values': {feature: float(shap_explanation[feature]) for feature in key_factors}
    }

# Generate force plot for visualization
def generate_shap_force_plot(application_data, model, scaler, explainer):
    # Prepare the input data (same as in predict_application)
    input_df = pd.DataFrame([application_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    for col in metadata['feature_names']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[metadata['feature_names']]
    
    numerical_cols = ['age', 'bmi', 'blood_pressure', 'income', 'credit_score', 'debt_to_income']
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = scaler.transform(input_df[[col]])
    
    # Get prediction
    prediction = model.predict(input_df)[0]
    pred_class_idx = list(model.classes_).index(prediction)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_df)
    
    # Create and save force plot
    shap.force_plot(
        explainer.expected_value[pred_class_idx], 
        shap_values[pred_class_idx], 
        input_df, 
        matplotlib=True,
        show=False,
        feature_names=metadata['feature_names']
    )
    plt.tight_layout()
    plt.savefig('static/shap_force_plot.png', bbox_inches='tight')
    plt.close()
    
    return 'static/shap_force_plot.png'

# Main execution
if __name__ == "__main__":
    print("Generating synthetic insurance application data...")
    insurance_data = generate_synthetic_data(2000)
    print(f"Generated {len(insurance_data)} synthetic applications")
    print("\nData sample:")
    print(insurance_data.head())
    print("\nDecision distribution:")
    print(insurance_data['decision'].value_counts())
    
    print("\nTraining underwriting model with SHAP explanations...")
    model, scaler, explainer = train_underwriting_model(insurance_data)
    
    # Example of using the model for a new application
    print("\nTesting model with a sample application...")
    sample_application = {
        'age': 42,
        'gender': 'M',
        'bmi': 28.5,
        'smoker': 0,
        'has_diabetes': 0,
        'has_heart_disease': 0,
        'blood_pressure': 125,
        'income': 85000,
        'credit_score': 720,
        'debt_to_income': 0.25
    }
    
    result = predict_application(sample_application, model, scaler, explainer)
    print("\nPrediction result with SHAP values:")
    print(json.dumps(result, indent=2))
    
    # Generate SHAP force plot for visualization
    force_plot_path = generate_shap_force_plot(sample_application, model, scaler, explainer)
    print(f"\nGenerated SHAP force plot at: {force_plot_path}")
    
    print("\nSaved model, explainer, and related files for API integration")