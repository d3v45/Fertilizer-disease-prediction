import google.generativeai as genai
import os

class AIAdvisor:
    def __init__(self):
        # 🔑 KEEP YOUR EXISTING KEY HERE
        self.api_key = "AIzaSyAibCPfdCfx6G6T5FsHrRk7moMbZVRZhtU" 
        
        self.client = False
        if "AIza" in self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.client = True
            except Exception as e:
                print(f"   [System] ⚠️ Gemini Error: {e}")

    def get_custom_advice(self, user_input, recommended_fert, crop):
        print("\n   [System] 🤖 Ai Advisor is thinking...")
        
        if not self.client:
            return "⚠️ Gemini Key missing."

        # UPDATED PROMPT: Added "Respond ONLY in English"
        prompt = f"""
        Act as an expert agricultural scientist.
        User location: {user_input.get('District_Name')}. Crop: {crop}.
        Soil: N={user_input.get('Nitrogen')}, P={user_input.get('Phosphorus')}, K={user_input.get('Potassium')}, pH={user_input.get('pH')}.
        
        Model Recommendation: {recommended_fert}.
        
        Task:
        1. Explain why this fertilizer is needed (Scientific reason).
        2. Suggest 2 organic alternatives.
        3. Provide a practical application schedule.
        
        IMPORTANT: Respond ONLY in English. Do not use local languages.
        Keep it short, professional, and practical.
        """

        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"