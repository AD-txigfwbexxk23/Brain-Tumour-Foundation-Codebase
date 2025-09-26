"""
Brain Tumor Prediction Model Inference Script
Professional Medical AI Assistant
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

class MultiOutputNN(nn.Module):
    """Multi-output neural network for brain tumor prediction"""
    def __init__(self, num_numeric, cat_dims, embedding_dims, hidden_dim=128):
        super(MultiOutputNN, self).__init__()
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(cat_dims[i], embedding_dims[i]) for i in range(len(cat_dims))])
        emb_total_dim = sum(embedding_dims)
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(num_numeric + emb_total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Heads
        self.survival_head = nn.Linear(hidden_dim // 2, 1)
        self.treatment_head = nn.Linear(hidden_dim // 2, 4)  # 4 treatment classes: None, Chemotherapy, Radiation, Surgery
    
    def forward(self, x_num, x_cat):
        x_emb = [emb(x_cat[:,i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, dim=1)
        x = torch.cat([x_num, x_emb], dim=1)
        x = self.shared(x)
        surv = self.survival_head(x)
        trt = self.treatment_head(x)
        return surv, trt

class BrainTumorPredictor:
    """Professional brain tumor prediction system"""
    
    def __init__(self, model_path="multi_output_brain_tumor_model.pth"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.cat_encoders = {}
        self.treatment_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature mappings
        self.growth_map = {'Slow': 0, 'Moderate': 1, 'Rapid': 2}
        self.severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
        
        # Expected categorical values
        self.expected_values = {
            'Tumor_Location': ['Cerebellum', 'Temporal', 'Occipital', 'Parietal', 'Frontal'],
            'MRI_Findings': ['Normal', 'Abnormal', 'Severe'],
            'Radiation_Exposure': ['Low', 'Medium', 'High']
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Load the model
            model_state = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model with correct dimensions
            # These values should match your training configuration
            num_numeric = 4  # Age, Tumor_Size, Tumor_Growth_Rate, Symptom_Severity
            cat_dims = [5, 3, 3]  # Tumor_Location, MRI_Findings, Radiation_Exposure
            embedding_dims = [3, 2, 2]  # Corresponding embedding dimensions
            
            self.model = MultiOutputNN(num_numeric, cat_dims, embedding_dims)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize scaler and encoders (these would need to be saved during training)
            # For now, we'll create them with expected values
            self._initialize_preprocessors()
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _initialize_preprocessors(self):
        """Initialize preprocessing objects with expected values"""
        # Initialize scaler for numeric features
        self.scaler = StandardScaler()
        # Fit with dummy data to initialize
        dummy_numeric = np.array([[50, 7.0, 1, 1]])  # Age, Tumor_Size, Growth_Rate, Severity
        self.scaler.fit(dummy_numeric)
        
        # Initialize categorical encoders
        for feature, values in self.expected_values.items():
            le = LabelEncoder()
            le.fit(values)
            self.cat_encoders[feature] = le
        
        # Initialize treatment encoder
        self.treatment_encoder = LabelEncoder()
        self.treatment_encoder.fit(['None', 'Chemotherapy', 'Radiation', 'Surgery'])
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame([input_data])
            
            # Map ordinal features
            df['Tumor_Growth_Rate'] = df['Tumor_Growth_Rate'].map(self.growth_map)
            df['Symptom_Severity'] = df['Symptom_Severity'].map(self.severity_map)
            
            # Prepare numeric features
            numeric_features = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate', 'Symptom_Severity']
            X_numeric = self.scaler.transform(df[numeric_features].values)
            
            # Prepare categorical features
            categorical_features = ['Tumor_Location', 'MRI_Findings', 'Radiation_Exposure']
            X_cat_indices = []
            
            for col in categorical_features:
                le = self.cat_encoders[col]
                X_cat_indices.append(le.transform(df[col].astype(str)))
            
            X_cat_indices = np.stack(X_cat_indices, axis=1)
            
            # Convert to tensors
            X_numeric_tensor = torch.tensor(X_numeric, dtype=torch.float32).to(self.device)
            X_cat_tensor = torch.tensor(X_cat_indices, dtype=torch.long).to(self.device)
            
            return X_numeric_tensor, X_cat_tensor
            
        except Exception as e:
            print(f"❌ Error preprocessing input: {e}")
            raise
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            # Preprocess input
            X_numeric, X_cat = self.preprocess_input(input_data)
            
            # Make prediction
            with torch.no_grad():
                surv_pred, trt_pred = self.model(X_numeric, X_cat)
                
                # Convert predictions
                survival_rate = surv_pred.cpu().numpy()[0][0] * 100  # Convert to percentage
                treatment_idx = torch.argmax(trt_pred, dim=1).cpu().numpy()[0]
                treatment = self.treatment_encoder.inverse_transform([treatment_idx])[0]
                
                # Calculate confidence for treatment
                treatment_probs = torch.softmax(trt_pred, dim=1).cpu().numpy()[0]
                treatment_confidence = treatment_probs[treatment_idx] * 100
                
                return {
                    'survival_rate': round(survival_rate, 2),
                    'treatment': treatment,
                    'treatment_confidence': round(treatment_confidence, 2),
                    'all_treatment_probs': {
                        treatment: round(prob * 100, 2) 
                        for treatment, prob in zip(self.treatment_encoder.classes_, treatment_probs)
                    }
                }
                
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def validate_input(self, input_data):
        """Validate input data"""
        errors = []
        
        # Validate numeric fields
        if not (0 <= input_data['Age'] <= 120):
            errors.append("Age must be between 0 and 120")
        
        if not (0 < input_data['Tumor_Size'] <= 20):
            errors.append("Tumor size must be between 0 and 20 cm")
        
        # Validate categorical fields
        if input_data['Tumor_Location'] not in self.expected_values['Tumor_Location']:
            errors.append(f"Tumor location must be one of: {', '.join(self.expected_values['Tumor_Location'])}")
        
        if input_data['MRI_Findings'] not in self.expected_values['MRI_Findings']:
            errors.append(f"MRI findings must be one of: {', '.join(self.expected_values['MRI_Findings'])}")
        
        if input_data['Radiation_Exposure'] not in self.expected_values['Radiation_Exposure']:
            errors.append(f"Radiation exposure must be one of: {', '.join(self.expected_values['Radiation_Exposure'])}")
        
        return errors

# Example usage
if __name__ == "__main__":
    predictor = BrainTumorPredictor()
    
    # Example input
    sample_input = {
        'Age': 45,
        'Tumor_Size': 6.5,
        'Tumor_Growth_Rate': 'Moderate',
        'Symptom_Severity': 'Moderate',
        'Tumor_Location': 'Frontal',
        'MRI_Findings': 'Abnormal',
        'Radiation_Exposure': 'Medium'
    }
    
    # Validate and predict
    errors = predictor.validate_input(sample_input)
    if errors:
        print("Validation errors:", errors)
    else:
        result = predictor.predict(sample_input)
        print("Prediction result:", result)
