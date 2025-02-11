
import os
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.disease_predictor import EnsembleModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def train():
    try:
        print("Loading data...")
        # Load and prepare data
        data = pd.read_csv('data/processed/disease_symptoms.csv')
        
        # Separate features and target
        X = data.drop('Disease', axis=1).astype(float).values
        
        # Encode disease labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['Disease'].values)
        
        print(f"Original data shape: {X.shape}")
        
        # Get feature names (symptoms)
        feature_names = data.drop('Disease', axis=1).columns.tolist()
        
        # Create and train model
        print("Initializing model...")
        model = EnsembleModel(
            input_size=X.shape[1],
            num_classes=len(label_encoder.classes_)
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of disease classes: {len(label_encoder.classes_)}")
        
        # Train model
        print("Training model...")
        model.train(X_train, y_train, feature_names)
        
        # Save classes
        model.classes_ = label_encoder.classes_
        
        # Save model
        print("Saving model...")
        model.save('models')
        
        print("Model training completed successfully!")
        
        # Print some statistics
        print("\nDisease distribution:")
        for idx, disease in enumerate(label_encoder.classes_):
            count = np.sum(y == idx)
            print(f"{disease}: {count} samples")
        
        return model, label_encoder
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


def print_symptom_analysis(model):
    """Print analysis of available symptoms"""
    print("\nAvailable Symptoms in Training Data:")
    valid_symptoms = sorted(model.get_all_symptoms())
    for symptom in valid_symptoms:
        print(f"- {symptom}")
    
    return valid_symptoms

def get_matching_symptoms(target_disease, valid_symptoms):
    """Get symptoms that match the training data"""
    matching_symptoms = []
    for symptom in valid_symptoms:
        
        if any(keyword in symptom for keyword in ['fever', 'cough', 'pain', 'fatigue', 'nausea']):
            matching_symptoms.append(symptom)
    return matching_symptoms

def test_model(model, label_encoder):
    print("\nTesting model with sample cases...")
    
    # Get and print valid symptoms
    valid_symptoms = print_symptom_analysis(model)
    
    
    test_cases = [
        {
            "symptoms": [
                "high_fever",
                "cough",
                "fatigue",
                "weight_loss",
                "loss_of_appetite",
                "chest_pain",
                "blood_in_sputum"
            ],
            "expected": "Tuberculosis"
        },
        {
            "symptoms": [
                "headache",
                "nausea",
                "vomiting",
                "blurred_and_distorted_vision",
                "pain_behind_the_eyes",
                "dizziness"
            ],
            "expected": "Migraine"
        },
        {
            "symptoms": [
                "chest_pain",
                "breathlessness",
                "sweating",
                "nausea",
                "vomiting",
                "fast_heart_rate"
            ],
            "expected": "Heart attack"
        },
        {
            "symptoms": [
                "high_fever",
                "headache",
                "nausea",
                "vomiting",
                "muscle_pain",
                "joint_pain",
                "fatigue",
                "skin_rash"
            ],
            "expected": "Dengue"
        },
        {
            "symptoms": [
                "continuous_sneezing",
                "runny_nose",
                "cough",
                "mild_fever",
                "fatigue",
                "throat_irritation"
            ],
            "expected": "Common Cold"
        }
    ]
    
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        predictions = model.predict(case["symptoms"])
        print(f"\nSymptoms: {', '.join(case['symptoms'])}")
        print(f"Expected: {case['expected']}")
        print("Top 3 predictions:")
        for pred in predictions:
            print(f"  {pred['disease']}: {pred['probability']:.4f} ({pred['confidence_level']})")
            if pred['probability'] > 0.1:  
                matching_symptoms = set(case['symptoms']).intersection(set(pred['symptoms']))
                if matching_symptoms:
                    print(f"  Matching symptoms: {', '.join(matching_symptoms)}")
        
        if predictions[0]['disease'] == case['expected']:
            correct += 1
            print("✓ Correct prediction!")
        else:
            print("✗ Incorrect prediction")
    
    accuracy = correct / total
    print(f"\nTest accuracy: {accuracy:.2f}")

    # Print most common symptoms for each disease
    print("\nCommon Symptoms by Disease:")
    for disease in model.classes_:
        print(f"\n{disease}:")
        disease_data = pd.read_csv('data/processed/disease_symptoms.csv')
        disease_samples = disease_data[disease_data['Disease'] == disease]
        symptoms = []
        for column in disease_samples.columns[:-1]:  # Exclude the Disease column
            if disease_samples[column].sum() > len(disease_samples)/2:  # More than 50% of cases
                symptoms.append(column)
        if symptoms:
            print(f"Common symptoms: {', '.join(symptoms)}")
def main():
    try:
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        
        # Train the model
        model, label_encoder = train()
        
        # Test the model
        test_model(model, label_encoder)
        
        # Print additional analysis
        print("\nModel Analysis:")
        print(f"Number of symptoms: {len(model.get_all_symptoms())}")
        print(f"Number of diseases: {len(model.classes_)}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)
if __name__ == "__main__":
    main()