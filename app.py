from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import warnings
import sys
import io
import base64
from contextlib import redirect_stdout

# Import ALL your existing modules
# Make sure these files are in the same folder as app.py
from crop_engine import CropRecommender
from dosage_engine import DosageCalculator
from weather_engine import WeatherService
from ai_advisor import AIAdvisor
from visualizer import visualize_impact
from xai_engine import explain_prediction
from main import FertilizerRecommender 

app = Flask(__name__, template_folder='.')
CORS(app)
warnings.filterwarnings('ignore')

# Initialize All Engines
fert_engine = FertilizerRecommender()
fert_engine.load_model()
crop_engine = CropRecommender()
crop_engine.load_model()
dosage_calc = DosageCalculator()
weather_bot = WeatherService()
ai_bot = AIAdvisor() 

# DATA: Static Fallback Guide (Copied from main.py)
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # --- LOG CAPTURE START ---
    capture_buffer = io.StringIO()
    
    # Variables to hold return data
    image_base64 = None
    diagnosis = "General Assessment"

    try:
        with redirect_stdout(capture_buffer):
            print(f"\n[Web Request] Processing data for {data.get('District_Name')}...")
            
            # 1. Live Weather
            temp, rain, warning = weather_bot.get_live_weather(data['District_Name'])
            print(f"   [Weather] Temp: {temp}°C, Rain: {rain}mm")
            
            # 2. Input Setup
            user_input = {
                'District_Name': data['District_Name'],
                'Crop': data['Crop'],
                'Soil_color': data['Soil_color'],
                'Nitrogen': float(data['Nitrogen']),
                'Phosphorus': float(data['Phosphorus']),
                'Potassium': float(data['Potassium']),
                'pH': float(data['pH']),
                'Rainfall': rain, 
                'Temperature': temp
            }

            # 3. Predictions
            is_suitable, best_crop = crop_engine.predict_suitability(user_input)
            print(f"   [Crop Engine] Suitability check: {is_suitable}")
            
            recommended_fertilizer = fert_engine.get_recommendation(user_input)
            print(f"   [Fertilizer Engine] Recommendation: {recommended_fertilizer}")
            
            qty, logic_msg = dosage_calc.calculate_dosage(
                data['Crop'], user_input, recommended_fertilizer
            )
            print(f"   [Dosage Engine] Calculated: {qty} kg/acre")
            
            # Lookup Diagnosis
            guide = FARMER_GUIDE.get(recommended_fertilizer, FARMER_GUIDE['General'])
            diagnosis = guide['issue']

            # 4. Visualizations
            visualize_impact(user_input, recommended_fertilizer)
            
            # Encode Image to Base64 to send to UI
            try:
                with open("impact_analysis.png", "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                print(f"   [Error] Image encoding failed: {e}")

            # Prepare data for SHAP (Backround process)
            input_df = pd.DataFrame([user_input])
            if 'Soil_color' in input_df.columns: 
                input_df['Soil_color'] = input_df['Soil_color'].str.strip()
            explain_prediction(fert_engine.pipeline, input_df)

            print("[System] Analysis Complete.")

    except Exception as e:
        print(f"[Error] {e}")
        # traceback helps see where the error is in the console
        import traceback
        traceback.print_exc()

    terminal_logs = capture_buffer.getvalue()
    # Print logs to the real terminal so you can see them too
    sys.__stdout__.write(terminal_logs)
    
    return jsonify({
        'weather': {'temp': temp, 'rain': rain, 'warning': warning},
        'suitability': {'is_suitable': bool(is_suitable), 'best_crop': best_crop},
        'fertilizer': {
            'name': recommended_fertilizer, 
            'dosage': qty, 
            'logic': logic_msg,
            'diagnosis': diagnosis
        },
        'impact_image': image_base64,
        'logs': terminal_logs
    })

# NEW ROUTE: Ask AI separately
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    data = request.json
    # Reconstruct input for AI
    user_input = {
        'District_Name': data['District_Name'],
        'Nitrogen': float(data['Nitrogen']),
        'Phosphorus': float(data['Phosphorus']),
        'Potassium': float(data['Potassium']),
        'pH': float(data['pH'])
    }
    advice = ai_bot.get_custom_advice(user_input, data['Fertilizer'], data['Crop'])
    return jsonify({'advice': advice})

if __name__ == '__main__':
    print("server starting...")
    app.run(debug=True, port=5000)