import pandas as pd
import os
import joblib
import sys
import warnings 

# IMPORT THE NEW MODULE
from crop_engine import CropRecommender

# ==========================================
# 0. SYSTEM SETUP
# ==========================================
warnings.filterwarnings('ignore') 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = "Crop_and_fertilizer_cleaned.csv"
FERT_MODEL_PATH = "fertilizer_model.joblib"

# FARMER GUIDE
FARMER_GUIDE = {
    'Urea': {
        'issue': 'Low Nitrogen (Yellow Leaves)',
        'why': 'Heavy rains may have washed away nutrients, or the soil has been used too much without rest.',
        'prevent': '1. Rotate crops with beans or legumes.\n   2. Add cow dung or compost before planting.'
    },
    'DAP': {
        'issue': 'Low Phosphorus (Weak Roots)',
        'why': 'Soil is naturally low in minerals or acidic soil is locking up the nutrients.',
        'prevent': '1. Use organic manure regularly.\n   2. Avoid over-watering which damages roots.'
    },
    'MOP': {
        'issue': 'Low Potassium (Poor Growth)',
        'why': 'Sandy soil often loses potassium quickly.',
        'prevent': '1. Add wood ash to the soil.\n   2. Use mulch to keep the soil moist and healthy.'
    },
    '19:19:19 NPK': {
        'issue': 'General Nutrient Balance Needed',
        'why': 'The crop needs a boost of all major nutrients for better yield.',
        'prevent': '1. Test soil regularly.\n   2. Maintain a balance of organic and chemical fertilizers.'
    },
    'General': {
        'issue': 'Multiple Nutrient Deficiencies',
        'why': 'The soil needs a specific mix to support the chosen crop.',
        'prevent': '1. Follow a strict crop rotation schedule.\n   2. Perform a lab soil test every 2 years.'
    }
}

class FertilizerRecommender:
    def __init__(self):
        self.pipeline = None
        self.numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
        self.categorical_features = ['District_Name', 'Soil_color', 'Crop']

    def train_model(self):
        print("   [System] 🧠 Training Fertilizer Model...")
        if not os.path.exists(DATA_PATH):
            print(f"❌ Error: '{DATA_PATH}' not found.")
            sys.exit(1)

        df = pd.read_csv(DATA_PATH)
        X = df.drop(['Fertilizer', 'Link'], axis=1, errors='ignore')
        y = df['Fertilizer']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])
        
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, FERT_MODEL_PATH)

    def load_model(self):
        if os.path.exists(FERT_MODEL_PATH):
            self.pipeline = joblib.load(FERT_MODEL_PATH)
        else:
            self.train_model()

    def get_recommendation(self, inputs):
        input_df = pd.DataFrame([inputs])
        if 'Soil_color' in input_df.columns:
            input_df['Soil_color'] = input_df['Soil_color'].str.strip()
        return self.pipeline.predict(input_df)[0]

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Initialize Engines
    fert_engine = FertilizerRecommender()
    fert_engine.load_model()
    
    crop_engine = CropRecommender()
    crop_engine.load_model()

    # 2. User Input
    user_input = {
        'District_Name': 'Kolhapur', 
        'Crop':          'Rice',     
        'Soil_color':    'Black',
        'Nitrogen':      20, 
        'Phosphorus':    50, 
        'Potassium':     50, 
        'pH':            6.5, 
        'Rainfall':      500, 
        'Temperature':   30
    }

    # 3. Get Intelligence
    is_suitable, best_crop = crop_engine.predict_suitability(user_input)
    recommended_fertilizer = fert_engine.get_recommendation(user_input)
    advice = FARMER_GUIDE.get(recommended_fertilizer, FARMER_GUIDE['General'])

    # 4. Generate Report
    print("\n" + "="*60)
    print(f"🌾  FARM ADVISORY REPORT")
    print("="*60)

    # Suitability Section
    user_crop = user_input['Crop']
    print(f"\n🔍  CROP SUITABILITY CHECK: '{user_crop}'")
    
    if is_suitable:
        print(f"    ✅ EXCELLENT CHOICE! '{user_crop}' is highly suitable for this soil.")
    else:
        print(f"    ⚠️  RISK ALERT: '{user_crop}' might struggle here.")
        print(f"    🌟  BETTER OPTION: Consider growing '{best_crop}' instead.")

    # Fertilizer Section
    print("-" * 60)
    print(f"💊  RECOMMENDED FERTILIZER:  {recommended_fertilizer}")
    print(f"📋  DIAGNOSIS:               {advice['issue']}")
    
    print("\n🤔  WHY THIS HAPPENED?")
    print(f"    {advice['why']}")
    
    print("\n🛡️   PREVENTION & CARE:")
    print(f"    {advice['prevent']}")
    print("\n" + "="*60)