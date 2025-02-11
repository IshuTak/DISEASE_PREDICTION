
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
from tqdm import tqdm
import json


SYMPTOM_MAPPING = {
    
    'fever': 'high_fever',
    'mild_fever': 'mild_fever',
    'high_fever': 'high_fever',
    
    
    'cough': 'continuous_cough',
    'continuous_cough': 'continuous_cough',
    
    
    'chest_pain': 'chest_pain',
    'headache': 'headache',
    'stomach_pain': 'stomach_pain',
    'abdominal_pain': 'abdominal_pain',
    'muscle_pain': 'muscle_pain',
    'joint_pain': 'joint_pain',
    
    
    'breathlessness': 'breathlessness',
    'difficulty_breathing': 'breathlessness',
    'shortness_of_breath': 'breathlessness',
    
    
    'fatigue': 'fatigue',
    'weakness': 'weakness',
    'tired': 'fatigue',
    'exhaustion': 'fatigue',
    
    'nausea': 'nausea',
    'vomiting': 'vomiting',
    'diarrhea': 'diarrhoea',
    
    'sweating': 'excessive_sweating',
    'loss_of_appetite': 'loss_of_appetite',
    'lack_of_appetite': 'loss_of_appetite',
    
    
    'dizziness': 'dizziness',
    'weight_loss': 'weight_loss',
    'restlessness': 'restlessness',
    'lethargy': 'lethargy',
    'irregular_sugar_level': 'irregular_sugar_level',
    'blurred_vision': 'blurred_vision',
    'excessive_hunger': 'excessive_hunger',
    'increased_appetite': 'excessive_hunger',
    'yellowing_eyes': 'yellowing_of_eyes',
    'yellow_eyes': 'yellowing_of_eyes',
    'yellow_skin': 'yellowing_of_skin'
}

DISEASE_SYMPTOMS = {
    'Tuberculosis': [
        'high_fever',
        'continuous_cough',
        'fatigue',
        'weight_loss',
        'loss_of_appetite',
        'chest_pain',
        'excessive_sweating'
    ],
    
    'Migraine': [
        'headache',
        'nausea',
        'vomiting',
        'blurred_vision',
        'sensitivity_to_light',
        'dizziness'
    ],
    
    'Heart attack': [
        'chest_pain',
        'breathlessness',
        'excessive_sweating',
        'vomiting',
        'nausea'
    ],
    
    'Common Cold': [
        'continuous_cough',
        'mild_fever',
        'fatigue',
        'headache',
        'runny_nose',
        'sneezing'
    ],
    
    'Pneumonia': [
        'high_fever',
        'breathlessness',
        'continuous_cough',
        'chest_pain',
        'fatigue',
        'vomiting'
    ],
    
    'Dengue': [
        'high_fever',
        'headache',
        'joint_pain',
        'muscle_pain',
        'nausea',
        'vomiting',
        'fatigue'
    ]
}

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.layer3 = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size//2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        return x

class EnsembleModel:
    def __init__(self, input_size, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        self.nn_model = SimpleNet(
            input_size=input_size,
            hidden_size=256,
            num_classes=num_classes
        ).to(self.device)
        
        self.scaler = StandardScaler()
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize feature names and classes
        self.feature_names = None
        self.classes_ = None
        
        # Store symptom mappings
        self.symptom_mapping = SYMPTOM_MAPPING
        self.disease_symptoms = DISEASE_SYMPTOMS

    def get_confidence_level(self, probability):
        """Convert probability to confidence level"""
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"

    def preprocess_symptoms(self, symptoms):
        """Convert symptoms list to feature vector"""
        if self.feature_names is None:
            raise ValueError("Model not trained. Feature names not available.")
            
        # Create feature vector
        features = np.zeros(len(self.feature_names))
        
        # Map and standardize symptoms
        mapped_symptoms = []
        for symptom in symptoms:
            # Clean the symptom text
            cleaned = symptom.lower().strip().replace(' ', '_')
            # Map to standard symptom if exists
            mapped = self.symptom_mapping.get(cleaned, cleaned)
            mapped_symptoms.append(mapped)
        
        # Set features for present symptoms
        for symptom in mapped_symptoms:
            if symptom in self.feature_names:
                idx = self.feature_names.index(symptom)
                features[idx] = 1
            else:
                print(f"Info: Symptom '{symptom}' not found in training data")
        
        return features.reshape(1, -1)

    def train(self, X, y, feature_names):
        """Train both RF and NN models"""
        try:
            # Store feature names
            self.feature_names = feature_names
            
            # Ensure data types are correct
            X = X.astype(np.float32)
            y = y.astype(np.int64)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest
            print("\nTraining Random Forest...")
            self.rf_model.fit(X_scaled, y)
            
            # Train Neural Network
            print("\nTraining Neural Network...")
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=True
            )
            
            self.nn_model.train()
            epochs = 30
            
            for epoch in tqdm(range(epochs), desc="Training NN"):
                for inputs, labels in dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.nn_model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
        except Exception as e:
            print(f"Error in training: {e}")
            raise

    # In src/models/disease_predictor.py

    def predict(self, symptoms):
        """Make predictions using both models"""
        
        try:
            print(f"Processing symptoms in model: {symptoms}")
            
            # Preprocess symptoms
            X = self.preprocess_symptoms(symptoms)
            X_scaled = self.scaler.transform(X)
            
            # Get RF predictions
            rf_proba = self.rf_model.predict_proba(X_scaled)
            
            # Get NN predictions
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            self.nn_model.eval()
            with torch.no_grad():
                nn_outputs = self.nn_model(X_tensor)
                nn_proba = torch.softmax(nn_outputs, dim=1).cpu().numpy()
            
            # Combine predictions
            final_proba = 0.6 * rf_proba + 0.4 * nn_proba
            
            # Get predictions with probabilities
            predictions = [(disease, prob) for disease, prob in 
                        zip(self.classes_, final_proba[0])]
            
            # Sort by probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Load disease info
            try:
                with open('data/processed/disease_info.json', 'r') as f:
                    disease_info = json.load(f)
                    print("Loaded disease info successfully")
            except Exception as e:
                print(f"Error loading disease info: {e}")
                disease_info = {}
            
            # Format top 3 predictions
            results = []
            for disease, probability in predictions[:3]:
                # Get disease information
                disease_data = disease_info.get(disease, {})
                print(f"Disease data for {disease}:", disease_data)
                
                # Map the JSON fields correctly based on your structure
                result = {
                    'disease': disease,
                    'probability': float(probability),
                    'confidence_level': self.get_confidence_level(probability),
                    'symptoms': disease_data.get('symptoms', []),
                    'description': disease_data.get('description', []),  # Array of precautions
                    'precautions': disease_data.get('precautions', [])  # Array of precautions
                }
                print(f"Formatted result:", result)
                results.append(result)
            
            print("Final results:", results)
            return results
                
        except Exception as e:
            print(f"Error in predict method: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save(self, path):
        """Save both models"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save RF model, scaler, and metadata
            joblib.dump({
                'rf_model': self.rf_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'classes': self.classes_
            }, f'{path}/rf_model.joblib')
            
            # Save NN model
            torch.save(self.nn_model.state_dict(), f'{path}/nn_model.pth')
            
            print(f"Models saved successfully to {path}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
            raise

    def load(self, path):
        """Load both models"""
        try:
            # Load RF model and metadata
            data = joblib.load(f'{path}/rf_model.joblib')
            self.rf_model = data['rf_model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.classes_ = data['classes']
            
            # Load NN model
            self.nn_model.load_state_dict(
                torch.load(f'{path}/nn_model.pth', map_location=self.device)
            )
            
            print(f"Models loaded successfully from {path}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def get_valid_symptoms_for_disease(self, disease):
        """Get valid symptoms for a specific disease"""
        return self.disease_symptoms.get(disease, [])

    def print_valid_symptoms(self):
        """Print all valid symptoms"""
        print("\nValid symptoms:")
        for symptom in sorted(self.feature_names):
            print(f"- {symptom}")

    def get_all_symptoms(self):
        """Return list of all valid symptoms"""
        return self.feature_names

    def get_disease_info(self, disease):
        """Get information about a specific disease"""
        symptoms = self.get_valid_symptoms_for_disease(disease)
        return {
            'disease': disease,
            'symptoms': symptoms,
            'description': self._get_disease_description(disease)
        }

    def _get_disease_description(self, disease):
        """Get description for a disease"""
        descriptions = {
            'Tuberculosis': 'A bacterial infection primarily affecting the lungs.',
            'Migraine': 'A severe headache often accompanied by sensitivity to light and nausea.',
            'Heart attack': 'A serious condition where blood flow to the heart is blocked.',
            'Common Cold': 'A viral infection affecting the upper respiratory tract.',
            'Pneumonia': 'An infection causing inflammation of the air sacs in the lungs.',
            'Dengue': 'A mosquito-borne viral infection causing severe flu-like illness.'
        }
        return descriptions.get(disease, 'No description available.')
    
    def get_disease_specific_symptoms(self, disease):
        """Get common symptoms for a specific disease based on training data"""
        try:
            # Define the most common symptoms for each disease
            disease_symptoms = {
                'Depression': [
                    'depression',
                    'mood_swings',
                    'anxiety',
                    'fatigue',
                    'loss_of_appetite',
                    'weight_gain',
                    'lack_of_concentration'
                ],
                'Anxiety': [
                    'anxiety',
                    'mood_swings',
                    'restlessness',
                    'fatigue',
                    'palpitations',
                    'sweating'
                ],
                'Obesity': [
                    'obesity',
                    'weight_gain',
                    'excessive_hunger',
                    'fatigue',
                    'increased_appetite'
                ],
                'Diabetes': [
                    'excessive_hunger',
                    'increased_appetite',
                    'fatigue',
                    'weight_loss',
                    'obesity',
                    'irregular_sugar_level'
                ],
                'Hypothyroidism': [
                    'weight_gain',
                    'fatigue',
                    'mood_swings',
                    'depression',
                    'abnormal_menstruation',
                    'lethargy'
                ]
                # Add more diseases and their common symptoms
            }
            
            return disease_symptoms.get(disease, [])
            
        except Exception as e:
            print(f"Error getting disease symptoms: {e}")
            return []