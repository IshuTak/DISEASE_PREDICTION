
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

class DataProcessor:
    def __init__(self):
        """Initialize DataProcessor with required data files"""
        self.base_dir = Path(__file__).parent.parent.parent
        self.model_dir = self.base_dir / 'models'
        self.data_dir = self.base_dir / 'data'
        
        # Load symptom binarizer
        self.symptom_binarizer = self._load_symptom_binarizer()
        
        # Load disease info
        self.disease_info = self._load_disease_info()
        
        # Load symptom severity
        self.symptom_severity = self._load_symptom_severity()

    def _load_symptom_binarizer(self) -> MultiLabelBinarizer:
        """Load the trained symptom binarizer"""
        binarizer_path = self.model_dir / 'symptom_binarizer.joblib'
        try:
            return joblib.load(binarizer_path)
        except Exception as e:
            print(f"Warning: Could not load symptom binarizer: {e}")
            return MultiLabelBinarizer()

    def _load_disease_info(self) -> Dict:
        """Load disease information from JSON"""
        info_path = self.data_dir / 'processed' / 'disease_info.json'
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load disease info: {e}")
            return {}

    def _load_symptom_severity(self) -> Dict[str, int]:
        """Load symptom severity data"""
        severity_path = self.data_dir / 'raw' / 'Symptom-severity.csv'
        try:
            df = pd.read_csv(severity_path)
            return dict(zip(
                df['Symptom'].str.lower().str.replace(' ', '_'),
                df['weight']
            ))
        except Exception as e:
            print(f"Warning: Could not load symptom severity: {e}")
            return {}

    def process_symptoms(self, symptoms: List[str]) -> np.ndarray:
        """Convert symptoms list to binary vector"""
        if not symptoms:
            return np.zeros((1, len(self.symptom_binarizer.classes_)))
        
        # Clean symptoms
        cleaned_symptoms = [
            s.lower().replace(' ', '_') for s in symptoms
        ]
        
        # Transform to binary vector
        try:
            return self.symptom_binarizer.transform([cleaned_symptoms])
        except Exception as e:
            print(f"Error processing symptoms: {e}")
            return np.zeros((1, len(self.symptom_binarizer.classes_)))

    def get_disease_info(self, disease: str) -> Dict[str, Any]:
        """Get information about a specific disease"""
        return self.disease_info.get(disease, {
            'description': 'No information available',
            'symptoms': [],
            'precautions': []
        })

    def calculate_severity_score(self, symptoms: List[str]) -> float:
        """Calculate severity score for given symptoms"""
        if not symptoms or not self.symptom_severity:
            return 0.0
        
        total_severity = sum(
            self.symptom_severity.get(s.lower().replace(' ', '_'), 0)
            for s in symptoms
        )
        
        max_severity = max(self.symptom_severity.values()) * len(symptoms)
        return total_severity / max_severity if max_severity > 0 else 0.0

    def validate_symptoms(self, symptoms: List[str]) -> Tuple[bool, str]:
        """Validate the provided symptoms"""
        if not symptoms:
            return False, "No symptoms provided"
        
        if len(symptoms) > 17:  # Maximum symptoms in dataset
            return False, "Too many symptoms provided"
        
        valid_symptoms = set(self.symptom_binarizer.classes_)
        invalid_symptoms = [
            s for s in symptoms 
            if s.lower().replace(' ', '_') not in valid_symptoms
        ]
        
        if invalid_symptoms:
            return False, f"Invalid symptoms: {', '.join(invalid_symptoms)}"
        
        return True, ""

    def get_all_symptoms(self) -> List[str]:
        """Get list of all valid symptoms"""
        return sorted(self.symptom_binarizer.classes_)
    
    def standardize_symptom(self, symptom: str) -> str:
        """Standardize symptom text to match training data format"""
        # Convert to lowercase and replace spaces with underscores
        symptom = symptom.lower().replace(' ', '_')
    
        
        symptom_mapping = {
            'cough': 'continuous_cough',
            'fever': 'high_fever',
            'tired': 'fatigue',
            'chest pain': 'chest_pain',
            'difficulty breathing': 'breathlessness',
            'throwing up': 'vomiting',
            'stomach pain': 'abdominal_pain',
            'head pain': 'headache',
            'dizzy': 'dizziness',
            'exhausted': 'fatigue',
            'cant breathe': 'breathlessness',
            'heart racing': 'fast_heart_rate',
            'feeling weak': 'weakness',
            'no appetite': 'loss_of_appetite',
            'feeling sick': 'nausea',
            'sweating a lot': 'sweating',
        
        }
    
        return symptom_mapping.get(symptom, symptom)

    def process_symptoms(self, symptoms: List[str]) -> np.ndarray:
        """Convert symptoms list to binary vector"""
        if not symptoms:
            return np.zeros((1, len(self.symptom_binarizer.classes_)))
        
        # Standardize symptoms
        cleaned_symptoms = [
            self.standardize_symptom(s) for s in symptoms
        ]
        
        # Filter out unknown symptoms
        valid_symptoms = [
            s for s in cleaned_symptoms 
            if s in self.symptom_binarizer.classes_
        ]
        
        if not valid_symptoms:
            print("Warning: No valid symptoms provided")
            return np.zeros((1, len(self.symptom_binarizer.classes_)))
        
        # Transform to binary vector
        try:
            return self.symptom_binarizer.transform([valid_symptoms])
        except Exception as e:
            print(f"Error processing symptoms: {e}")
            return np.zeros((1, len(self.symptom_binarizer.classes_)))

    def format_prediction_result(self, 
                               disease: str, 
                               probability: float, 
                               symptoms: List[str]) -> Dict[str, Any]:
        """Format the prediction result with additional information"""
        disease_info = self.get_disease_info(disease)
        severity_score = self.calculate_severity_score(symptoms)
        
        return {
            'disease': disease,
            'probability': float(probability),
            'severity_score': severity_score,
            'description': disease_info.get('description', ''),
            'precautions': disease_info.get('precautions', []),
            'common_symptoms': disease_info.get('symptoms', [])
        }