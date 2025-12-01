import requests

class WeatherService:
    def __init__(self):
        # 🔑 THIS IS YOUR VERIFIED WORKING KEY
        self.API_KEY = "3afe14c277214865f4f25b4953f18ad5"
        self.BASE_URL = "http://api.openweathermap.org/data/2.5/"

    def get_live_weather(self, district):
        """
        Returns: (Temperature, Rainfall_estimate, Warning_Message)
        """
        print(f"   [System] ☁️  Connecting to Weather Satellite for {district}...")
        
        try:
            # 1. Get Current Weather
            url = f"{self.BASE_URL}weather?q={district}&appid={self.API_KEY}&units=metric"
            response = requests.get(url, timeout=10).json()
            
            # 2. Check for Errors (Like 401 Invalid Key)
            if str(response.get('cod')) != "200":
                print(f"   [System] ⚠️ Weather API Error: {response.get('message')}")
                # Fallback to defaults if API fails
                return 25, 600, None

            # 3. Parse Data
            temp = response['main']['temp']
            
            # Rainfall: 'rain.1h' exists only if it is raining right now
            rain_1h = response.get('rain', {}).get('1h', 0)
            
            # Estimate Annual Rainfall (Heuristic: Current rain * 24h * 30d + Base)
            # This is a prediction logic for the project
            estimated_annual_rain = 800 + (rain_1h * 720) 

            # 4. Generate Warning
            warning = None
            if rain_1h > 2:
                warning = f"⚠️ IT IS RAINING ({rain_1h}mm)! Do not apply fertilizer."

            return temp, estimated_annual_rain, warning

        except Exception as e:
            print(f"   [Error] Weather Fetch Failed: {e}")
            return 25, 600, None