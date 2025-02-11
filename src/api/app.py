
import os
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.models.disease_predictor import EnsembleModel
import pandas as pd
import numpy as np

app = Flask(__name__, 
           template_folder=str(project_root / 'templates'),
           static_folder=str(project_root / 'static'))
CORS(app)

model = None

def init_model():
    """Initialize the model"""
    global model
    try:
        # Load processed data to get dimensions
        data_path = project_root / 'data' / 'processed' / 'disease_symptoms.csv'
        data = pd.read_csv(data_path)
        num_symptoms = len(data.columns) - 1  # Exclude Disease column
        num_diseases = len(data['Disease'].unique())
        
        # Initialize model
        model = EnsembleModel(
            input_size=num_symptoms,
            num_classes=num_diseases
        )
        
        # Load trained model
        model_path = project_root / 'models'
        model.load(str(model_path))
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initialize model when starting the app
init_model()

@app.route('/')
def home():
    """Render home page"""
    try:
        if model is None:
            if not init_model():
                return "Error: Model not initialized", 500
        
        # Get list of symptoms for the template
        symptoms = model.get_all_symptoms()
        return render_template('index.html', symptoms=sorted(symptoms))
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on symptoms"""
    try:
        # Check if model is initialized
        if model is None:
            if not init_model():
                return jsonify({'error': 'Model not initialized'}), 500
        
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
            
        symptoms = data['symptoms']
        print("Processing symptoms:", symptoms)  # Debug print
        
        if not symptoms:
            return jsonify({'error': 'Empty symptoms list'}), 400
            
        # Get predictions
        predictions = model.predict(symptoms)
        print("Predictions:", predictions)  # Debug print
        
        # Format response
        response = {
            'success': True,
            'predictions': predictions
        }
        
        print("Sending response:", response)  # Debug print
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full error traceback
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get list of all symptoms"""
    try:
        # Check if model is initialized
        if model is None:
            if not init_model():
                return jsonify({'error': 'Model not initialized'}), 500
            
        symptoms = model.get_all_symptoms()
        return jsonify({
            'success': True,
            'symptoms': sorted(symptoms)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/diseases', methods=['GET'])
def get_diseases():
    """Get list of all diseases"""
    try:
        # Check if model is initialized
        if model is None:
            if not init_model():
                return jsonify({'error': 'Model not initialized'}), 500
            
        diseases = sorted(model.classes_)
        return jsonify({
            'success': True,
            'diseases': diseases
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Check if the service is healthy"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Ensure directories exist
    for directory in ['templates', 'static', 'models', 'data/processed']:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist")
    
    # Ensure the model is initialized before starting the server
    if not model:
        print("Initializing model...")
        if not init_model():
            print("Failed to initialize model")
            sys.exit(1)
    
    print(f"Template directory: {app.template_folder}")
    print(f"Static directory: {app.static_folder}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)