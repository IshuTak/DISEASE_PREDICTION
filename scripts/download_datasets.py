
import os
import kaggle
from dotenv import load_dotenv

load_dotenv()

def download_kaggle_datasets():
    # Configure Kaggle
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
    
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Download disease-symptom dataset
    kaggle.api.dataset_download_files(
        'itachi9604/disease-symptom-description-dataset',
        path='data/raw',
        unzip=True
    )
    
    print("Datasets downloaded successfully!")

if __name__ == "__main__":
    download_kaggle_datasets()