import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import json
import joblib
import os

def download_disease_dataset():
    """
    Load disease dataset from local files
    """
    try:
        return {
            'dataset': pd.read_csv('data/raw/dataset.csv'),
            'severity': pd.read_csv('data/raw/Symptom-severity.csv'),
            'description': pd.read_csv('data/raw/symptom_Description.csv'),
            'precaution': pd.read_csv('data/raw/symptom_precaution.csv')
        }
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        raise

def clean_symptom(symptom):
    """Clean symptom text"""
    if pd.isna(symptom):
        return ''
    return str(symptom).strip().lower().replace(' ', '_')

def preprocess_disease_data(data_dict):
    """Process the disease dataset"""
    # Process main dataset
    df = data_dict['dataset']
    
    # Get all symptom columns
    symptom_cols = [col for col in df.columns if 'Symptom' in col]
    
    # Clean symptoms
    for col in symptom_cols:
        df[col] = df[col].apply(clean_symptom)
    
    # Combine symptoms into a list
    df['Symptoms'] = df.apply(
        lambda x: [s for s in x[symptom_cols] if s != ''], 
        axis=1
    )
    
    # Create binary matrix of symptoms
    mlb = MultiLabelBinarizer()
    symptom_matrix = mlb.fit_transform(df['Symptoms'])
    
    # Create processed dataset
    processed_df = pd.DataFrame(
        symptom_matrix,
        columns=mlb.classes_
    )
    processed_df['Disease'] = df['Disease']
    
    return processed_df, mlb

def create_disease_info_dict(data_dict):
    """Create a dictionary with detailed disease information"""
    disease_info = {}
    
    # Process main dataset
    for _, row in data_dict['dataset'].iterrows():
        disease = row['Disease']
        if disease not in disease_info:
            # Get symptoms excluding empty strings
            symptoms = [
                s for s in row[1:] 
                if isinstance(s, str) and s.strip() != ''
            ]
            
            disease_info[disease] = {
                'symptoms': symptoms,
                'description': '',
                'precautions': []
            }
    
    # Add descriptions
    if 'description' in data_dict:
        for _, row in data_dict['description'].iterrows():
            disease = row['Disease']
            if disease in disease_info:
                disease_info[disease]['description'] = row['Description']
    
    # Add precautions
    if 'precaution' in data_dict:
        for _, row in data_dict['precaution'].iterrows():
            disease = row['Disease']
            if disease in disease_info:
                # Get precautions excluding NaN values
                precautions = [
                    p for p in row[1:] 
                    if isinstance(p, str) and p.strip() != ''
                ]
                disease_info[disease]['precautions'] = precautions
    
    return disease_info

def save_processed_data(processed_df, disease_info, mlb):
    """Save processed data to files"""
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed dataset
    processed_df.to_csv('data/processed/disease_symptoms.csv', index=False)
    print("Saved processed symptoms data")
    
    # Save disease info
    with open('data/processed/disease_info.json', 'w') as f:
        json.dump(disease_info, f, indent=4)
    print("Saved disease information")
    
    # Save label binarizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(mlb, 'models/symptom_binarizer.joblib')
    print("Saved symptom binarizer")

def main():
    try:
        print("Loading datasets...")
        data_dict = download_disease_dataset()
        
        print("Processing disease-symptom data...")
        processed_df, mlb = preprocess_disease_data(data_dict)
        
        print("Creating disease information dictionary...")
        disease_info = create_disease_info_dict(data_dict)
        
        print("Saving processed data...")
        save_processed_data(processed_df, disease_info, mlb)
        
        print("Data processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()