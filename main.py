import pandas as pd
import os
import joblib
import sys
import warnings 

# ==========================================
# IMPORT LOCAL MODULES
# ==========================================
# Make sure you have created these 4 files in the same folder!
from crop_engine import CropRecommender
from visualizer import visualize_impact
from dosage_engine import DosageCalculator  # <--- New Feature 2
from xai_engine import explain_prediction   # <--- New Feature 3
from weather_engine import WeatherService   # <--- New Feature 4
from ai_advisor import AIAdvisor            # <--- New Feature 5

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

# STATIC FALLBACK GUIDE (Used if AI fails or for quick summary)
FARMER_GUIDE = {
    'Urea': {
        'issue': 'Low Nitrogen (Yellow Leaves)',
        'why': 'Heavy rains may have washed away nutrients.',
        'prevent': 'Rotate crops with beans or legumes.'
    },
    'DAP': {
        'issue': 'Low Phosphorus (Weak Roots)',
        'why': 'Soil is naturally low in minerals or acidic.',
        'prevent': 'Use organic manure regularly.'
    },
    'MOP': {
        'issue': 'Low Potassium (Poor Growth)',
        'why': 'Sandy soil often loses potassium quickly.',
        'prevent': 'Add wood ash to the soil.'
    },
    '19:19:19 NPK': {
        'issue': 'General Nutrient Balance Needed',
        'why': 'The crop needs a boost of all major nutrients.',
        'prevent': 'Test soil regularly.'
    },
    'General': {
        'issue': 'Multiple Nutrient Deficiencies',
        'why': 'The soil needs a specific mix.',
        'prevent': 'Perform a lab soil test every 2 years.'
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
    print("\n" + "="*60)
    print(f"🚜  SMART FARMING SYSTEM INITIALIZING...")
    print("="*60)

    # 1. Initialize All Engines
    fert_engine = FertilizerRecommender()
    fert_engine.load_model()
    
    crop_engine = CropRecommender()
    crop_engine.load_model()
    
    # New Engines (Ensure files exist!)
    try:
        dosage_calc = DosageCalculator()
        weather_bot = WeatherService()
        ai_bot = AIAdvisor()
    except NameError as e:
        print(f"\n❌ CRITICAL ERROR: Missing Module. {e}")
        print("   Please create dosage_engine.py, weather_engine.py, and ai_advisor.py")
        sys.exit(1)

    # 2. Context Gathering (Feature 4: Live Weather)
    target_district = 'Kolhapur'
    print(f"   [System] 📡 Fetching real-time data for {target_district}...")
    
    # We fetch rainfall/temp automatically. No need for manual input!
    live_temp, live_rain, rain_warning = weather_bot.get_live_weather(target_district)
    print(f"   [Weather] Temp: {live_temp}°C | Rain Estimate: {live_rain}mm")

    # 3. Build User Input (Auto-injecting weather data)
    user_input = {
        'District_Name': target_district, 
        'Crop':          'Rice',     
        'Soil_color':    'Black',
        'Nitrogen':      20, 
        'Phosphorus':    50, 
        'Potassium':     50, 
        'pH':            6.5, 
        # LIVE DATA INJECTION (Automatically overrides manual input)
        'Rainfall':      live_rain, 
        'Temperature':   live_temp   
    }

    # 4. Get Predictions
    is_suitable, best_crop = crop_engine.predict_suitability(user_input)
    recommended_fertilizer = fert_engine.get_recommendation(user_input)
    
    # Calculate Precision Dosage (Feature 2)
    qty_per_acre, logic_msg = dosage_calc.calculate_dosage(
        user_input['Crop'], user_input, recommended_fertilizer
    )

    # Get Static Guide (Fallback)
    advice = FARMER_GUIDE.get(recommended_fertilizer, FARMER_GUIDE['General'])

    # 5. Generate Comprehensive Report
    print("\n" + "="*60)
    print(f"🌾  FARM ADVISORY REPORT (v2.0)")
    print("="*60)

    # Weather Alert
    if rain_warning:
        print(f"\n🚨  CRITICAL WEATHER WARNING: {rain_warning}")
        print("    (Application of fertilizer is NOT recommended today)")

    # Section 1: Suitability
    user_crop = user_input['Crop']
    print(f"\n1️⃣  CROP SUITABILITY: '{user_crop}'")
    if is_suitable:
        print(f"    ✅ EXCELLENT CHOICE! Soil conditions are perfect.")
    else:
        # --- CHANGED LINE BELOW ---
        print(f"    🌟 RECOMMENDED CROP: Consider growing '{best_crop}' for better yield.")

    # Section 2: Fertilizer Plan
    print("-" * 60)
    print(f"2️⃣  NUTRIENT MANAGEMENT")
    print(f"    💊 Recommended:     {recommended_fertilizer}")
    print(f"    ⚖️  Dosage:          {qty_per_acre} kg / acre")
    print(f"    📝  Calculation:     {logic_msg}")
    print(f"    🩺  Diagnosis:       {advice['issue']}")

    # Section 3: AI Consultation (Feature 5)
    print("-" * 60)
    print(f"3️⃣  AI CONSULTANT (GenAI)")
    ai_advice = ai_bot.get_custom_advice(user_input, recommended_fertilizer, user_crop)
    print(ai_advice)

    # 6. Visualizations (Feature 3: XAI & Impact)
    print("\n" + "="*60)
    
    # Visual 1: Before vs After Bar Chart
    visualize_impact(user_input, recommended_fertilizer)
    
    # Visual 2: Why did the AI choose this? (SHAP)
    # Convert input to DataFrame for SHAP
    input_df = pd.DataFrame([user_input])
    if 'Soil_color' in input_df.columns: 
        input_df['Soil_color'] = input_df['Soil_color'].str.strip()
    
    explain_prediction(fert_engine.pipeline, input_df)
    
    print("\n✅ System Finished.")