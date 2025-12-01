import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# CONFIGURATION
DATA_PATH = "Crop_and_fertilizer_cleaned.csv"
CROP_MODEL_PATH = "crop_model.joblib"

class CropRecommender:
    def __init__(self):
        self.pipeline = None
        # Input features for Crop Prediction (WE DO NOT USE 'Crop' AS INPUT HERE)
        self.numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
        self.categorical_features = ['District_Name', 'Soil_color']

    def train_model(self):
        print("   [System] 🌽 Training Crop Recommendation Model...")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"❌ '{DATA_PATH}' not found. Run clean_dataset.py first!")

        df = pd.read_csv(DATA_PATH)
        
        # X = Environment Data, y = Best Crop
        X = df.drop(['Crop', 'Fertilizer', 'Link'], axis=1, errors='ignore')
        y = df['Crop']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, CROP_MODEL_PATH)
        print(f"   [System] 💾 Crop Model saved to '{CROP_MODEL_PATH}'")

    def load_model(self):
        if os.path.exists(CROP_MODEL_PATH):
            self.pipeline = joblib.load(CROP_MODEL_PATH)
        else:
            self.train_model()

    def predict_suitability(self, user_inputs):
        """
        Determines if the user's chosen crop is suitable.
        Returns: (is_suitable, best_crop_suggestion)
        """
        # Remove 'Crop' from inputs as the model doesn't use it for prediction
        input_df = pd.DataFrame([user_inputs]).drop(['Crop'], axis=1, errors='ignore')
        
        if 'Soil_color' in input_df.columns:
            input_df['Soil_color'] = input_df['Soil_color'].str.strip()
            
        predicted_best_crop = self.pipeline.predict(input_df)[0]
        
        user_crop = user_inputs.get('Crop', '')
        
        # Check if user's choice matches the model's recommendation
        is_suitable = (user_crop.lower() == predicted_best_crop.lower())
        
        return is_suitable, predicted_best_crop