import pandas as pd
import numpy as np
import os
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

# ==========================================
# 0. SYSTEM SETUP
# ==========================================
warnings.filterwarnings('ignore') 

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = "Crop_and_fertilizer_cleaned.csv"
MODEL_PATH = "fertilizer_model.joblib"
MATRIX_FILE = "confusion_matrix_heatmap.png"

# Knowledge Base
NUTRIENT_MAPPING = {
    'Urea': 'Nitrogen (N)',
    'DAP': 'Nitrogen (N) & Phosphorus (P)',
    'MOP': 'Potassium (K)',
    'SSP': 'Phosphorus (P)',
    'Magnesium Sulphate': 'Magnesium & Sulphur',
    'Ammonium Sulphate': 'Nitrogen & Sulphur',
    'Ferrous Sulphate': 'Iron (Fe)',
    'White Potash': 'Potassium (K)',
    'Sulphur': 'Sulphur (S)',
    'Hydrated Lime': 'Soil Acidity (Low pH)',
    'Chilated Micronutrient': 'Micronutrients (Zn, Fe, etc.)',
    '10:26:26 NPK': 'High Phosphorus & Potassium',
    '19:19:19 NPK': 'Balanced NPK',
    '20:20:20 NPK': 'Balanced NPK',
    '13:32:26 NPK': 'Phosphorus & Potassium',
    '12:32:16 NPK': 'Phosphorus & Potassium',
    '50:26:26 NPK': 'High Nitrogen',
    '18:46:00 NPK': 'High Phosphorus',
    '10:10:10 NPK': 'Balanced NPK'
}

class FertilizerRecommender:
    def __init__(self):
        self.pipeline = None
        self.numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
        self.categorical_features = ['District_Name', 'Soil_color', 'Crop']

    def train_model(self):
        print(f"\n[System] 🔄 Starting Training Process using {DATA_PATH}...")
        
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            print(f"❌ Error: '{DATA_PATH}' not found. Please run clean_data.py first!")
            sys.exit(1)

        X = df.drop(['Fertilizer', 'Link'], axis=1, errors='ignore')
        y = df['Fertilizer']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])

        # Optimized: 200 Trees + Balanced Weights
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, 
                                                  random_state=42, 
                                                  class_weight='balanced'))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        
        print("[System] ✅ Training Complete. Calculating Metrics...")
        train_acc = self.pipeline.score(X_train, y_train)
        test_acc = self.pipeline.score(X_test, y_test)
        y_pred = self.pipeline.predict(X_test)

        print("\n" + "="*40)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*40)
        print(f"   Training Accuracy:  {train_acc:.2%}")
        print(f"   Testing Accuracy:   {test_acc:.2%}")
        print("-" * 40)
        
        # Zero division fix for clean report
        print("\nDetailed Classification Report (F1 Score):")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        self.save_confusion_matrix(y_test, y_pred)
        joblib.dump(self.pipeline, MODEL_PATH)
        print(f"\n[System] 💾 Model saved to '{MODEL_PATH}'")

    def save_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.pipeline.named_steps['classifier'].classes_
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(MATRIX_FILE)
        print(f"[System] 🖼️  Confusion Matrix saved as '{MATRIX_FILE}'")

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            print(f"[System] 📂 Loading existing model from '{MODEL_PATH}'...")
            self.pipeline = joblib.load(MODEL_PATH)
        else:
            self.train_model()

    def get_recommendation(self, inputs):
        """Returns ONLY the single best prediction."""
        input_df = pd.DataFrame([inputs])
        if 'Soil_color' in input_df.columns:
            input_df['Soil_color'] = input_df['Soil_color'].str.strip()
            
        # Get Probabilities
        probs = self.pipeline.predict_proba(input_df)[0]
        classes = self.pipeline.named_steps['classifier'].classes_
        
        # Find the winner (Max probability)
        winner_idx = np.argmax(probs)
        winner_name = classes[winner_idx]
        winner_score = probs[winner_idx] * 100
        reason = NUTRIENT_MAPPING.get(winner_name, "General Nutrients")
            
        return winner_name, winner_score, reason

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    engine = FertilizerRecommender()
    
    # Check for model existence
    if os.path.exists(MODEL_PATH):
        print("[System] 🗑️  Removing old model to force re-training...")
        os.remove(MODEL_PATH)
        
    engine.load_model()

    # ---------------------------------------------------------
    # 📝 USER INPUTS
    # ---------------------------------------------------------
    user_input = {
        'District_Name': 'Kolhapur', 
        'Crop':          'Jowar', 
        'Soil_color':    'Black',
        'Nitrogen':      20, 
        'Phosphorus':    50, 
        'Potassium':     50, 
        'pH':            6.5, 
        'Rainfall':      500, 
        'Temperature':   30
    }
    # ---------------------------------------------------------

    print("\n" + "="*50)
    print("🌱 INPUT PARAMETERS")
    print("="*50)
    for key, val in user_input.items():
        print(f"   {key:<15}: {val}")

    # Get Single Best Recommendation
    pred, conf, reason = engine.get_recommendation(user_input)

    print("\n" + "="*50)
    print("🔬 DIAGNOSTIC REPORT")
    print("="*50)
    
    # Simplified Status
    if conf > 80: status = "HIGH"
    elif conf > 50: status = "MODERATE"
    else: status = "LOW"

    print(f"🏆 RECOMMENDATION:   {pred}")
    print(f"💊 REASON:           Deficiency in {reason}")
    print(f"📊 CONFIDENCE:       {conf:.1f}% [{status}]")
    print("="*50)