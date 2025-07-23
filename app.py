from flask import Flask, request, jsonify, send_from_directory, make_response, render_template, redirect, url_for
import pickle
import pandas as pd
import json
import os
import shap
import matplotlib.pyplot as plt
import uuid
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Path for decisions CSV file
DECISIONS_CSV_PATH = 'data/underwriting_decisions.csv'

# Check if CSV exists, create with headers if not
def initialize_csv():
    if not os.path.exists(DECISIONS_CSV_PATH):
        with open(DECISIONS_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'applicant_id', 'decision', 'age', 'gender', 'bmi', 
                'smoker', 'has_diabetes', 'has_heart_disease', 'blood_pressure', 
                'income', 'credit_score', 'debt_to_income',
                'override_decision', 'override_reason'
            ])

initialize_csv()

# Load the model and related files
@app.before_request
def load_model():
    global model, scaler, explainer, metadata
    if not hasattr(app, 'model_loaded'):
        model = pickle.load(open('underwriting_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        app.model_loaded = True

# API endpoint for underwriting decisions
@app.route('/api/underwrite', methods=['POST'])
def underwrite():
    # Get application data from request
    application_data = request.json
    
    # Validate required fields
    required_fields = ['age', 'gender', 'bmi', 'smoker', 'blood_pressure', 'income', 'credit_score']
    for field in required_fields:
        if field not in application_data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Prepare the input data
    input_df = pd.DataFrame([application_data])
    
    # Convert categorical variables to dummy variables
    input_df = pd.get_dummies(input_df, drop_first=True)
    
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
    sorted_shap = sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    key_factors = {k: v for k, v in sorted_shap[:5]}
    
    # Generate waterfall plot for this prediction
    plot_filename = f"shap_waterfall_{uuid.uuid4()}.png"
    plot_path = f"static/{plot_filename}"
    
    plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[pred_class_idx], 
        pred_shap_values,
        feature_names=input_df.columns.tolist(),
        max_display=10,
        show=False
    )
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # Format response
    response = {
        'applicant_id': application_data.get('applicant_id', 'unknown'),
        'decision': prediction,
        'confidence': confidence_scores,
        'key_factors': key_factors,
        'shap_values': {k: v for k, v in sorted_shap[:10]},  # Include more SHAP values for frontend viz
        'shap_base_value': float(explainer.expected_value[pred_class_idx]),
        'shap_plot_url': f"/static/{plot_filename}",
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Store decision for audit purposes
    store_decision(application_data, response)
    
    return jsonify(response)

# API endpoint for storing overrides
@app.route('/api/override', methods=['POST'])
def store_override():
    override_data = request.json
    
    required_fields = ['applicant_id', 'original_decision', 'override_decision', 'override_reason']
    for field in required_fields:
        if field not in override_data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Update CSV with override information
    with open(DECISIONS_CSV_PATH, 'r') as csvfile:
        rows = list(csv.reader(csvfile))
    
    updated = False
    for i in range(1, len(rows)):  # Skip header row
        if rows[i][1] == override_data['applicant_id']:  # Check applicant_id column
            # Add override decision and reason to the end
            if len(rows[i]) < 14:  # If override columns don't exist yet
                rows[i].extend(['', ''])  # Add empty columns
            rows[i][-2] = override_data['override_decision']
            rows[i][-1] = override_data['override_reason']
            updated = True
            break
    
    if updated:
        with open(DECISIONS_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        return jsonify({'status': 'success', 'message': 'Override recorded successfully'})
    else:
        return jsonify({'error': 'Applicant ID not found'}), 404

# Serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Simple in-memory storage for audit trail
decisions_log = []

def store_decision(application, decision):
    # Store in memory
    decision_record = {
        'application': application,
        'decision': decision['decision'],
        'key_factors': decision['key_factors'],
        'timestamp': decision['timestamp']
    }
    decisions_log.append(decision_record)
    
    # Store in CSV file
    with open(DECISIONS_CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().isoformat(),
            application.get('applicant_id', 'unknown'),
            decision['decision'],
            application.get('age', ''),
            application.get('gender', ''),
            application.get('bmi', ''),
            application.get('smoker', ''),
            application.get('has_diabetes', ''),
            application.get('has_heart_disease', ''),
            application.get('blood_pressure', ''),
            application.get('income', ''),
            application.get('credit_score', ''),
            application.get('debt_to_income', ''),
            '',  # placeholder for override_decision
            ''   # placeholder for override_reason
        ])

# API endpoint to retrieve audit log
@app.route('/api/audit', methods=['GET'])
def get_audit_log():
    return jsonify(decisions_log)

# API endpoint to download CSV
@app.route('/api/download-csv', methods=['GET'])
def download_csv():
    response = make_response(send_from_directory('data', 'underwriting_decisions.csv'))
    response.headers['Content-Disposition'] = 'attachment; filename=underwriting_decisions.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# Add route for landing page
@app.route('/')
def landing():
    return render_template('landing_page.html')

# Add route for demo page
@app.route('/demo')
def demo():
    return send_from_directory('.', 'index.html')

# Update the main route to redirect to landing page
@app.route('/index')
def index():
    return redirect(url_for('landing'))

# Contact form submission handler
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    
    # Here you would typically:
    # 1. Validate the form data
    # 2. Send an email notification
    # 3. Store the message in a database
    # 4. Return a success message
    
    return jsonify({
        'status': 'success',
        'message': 'Thank you for your message. We will get back to you soon!'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)