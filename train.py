import pandas as pd
import joblib
import sys
import os
import warnings

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# CONFIGURATION
RAW_DATA = "Crop and fertilizer dataset.csv"
CLEAN_DATA = "Crop_and_fertilizer_cleaned.csv"
MODEL_FILE = "fertilizer_model.joblib"

warnings.filterwarnings('ignore')

def get_data():
    """Smart Data Loader: Uses cleaned file if available, else cleans raw data."""
    
    # 1. Check if clean file already exists
    if os.path.exists(CLEAN_DATA):
        print(f"[1/3] 📂 Loading existing cleaned data: {CLEAN_DATA}")
        return pd.read_csv(CLEAN_DATA)

    # 2. If not, look for raw file
    print(f"[1/3] 🧹 Clean file not found. Cleaning {RAW_DATA}...")
    if not os.path.exists(RAW_DATA):
        print(f"❌ Error: '{RAW_DATA}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(RAW_DATA)
    
    # Clean it
    if 'Link' in df.columns: df.drop('Link', axis=1, inplace=True)
    if 'Soil_color' in df.columns: df['Soil_color'] = df['Soil_color'].str.strip()
    df.drop_duplicates(inplace=True)
    
    # Save it
    df.to_csv(CLEAN_DATA, index=False)
    print(f"   ✅ Data saved to '{CLEAN_DATA}'")
    return df

def train_network():
    """Trains the Random Forest Model."""
    print(f"[2/3] 🧠 Training Model...")
    
    # Load Data
    df = get_data()
    
    X = df.drop(['Fertilizer'], axis=1)
    y = df['Fertilizer']
    
    # Define Features
    numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
    categorical_features = ['District_Name', 'Soil_color', 'Crop']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Model Pipeline (Confirmed: class_weight IS balanced)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,          
            random_state=42,
            class_weight='balanced'    # <--- This fixes the bias
        ))
    ])
    
    # Stratified Split: Forces "Hard Exam" (Includes rare classes in test)
    # If you remove 'stratify=y', accuracy goes up but reliability goes down.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    print(f"[3/3] 📊 Evaluating Performance...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n   Training Accuracy: {pipeline.score(X_train, y_train):.2%}")
    print(f"   Testing Accuracy:  {acc:.2%} (Honest Score)")
    print("\n   --- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\n✅ SUCCESS: Model saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    train_network()