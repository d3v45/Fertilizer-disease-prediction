import requests

def test_weather_key():
    # The key you provided
    api_key = "3afe14c277214865f4f25b4953f18ad5"
    city = "Kolhapur"
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    print(f"Testing API Key: {api_key}")
    print(f"Target URL: {base_url}?q={city}&appid={api_key}&units=metric")
    print("-" * 40)

    try:
        url = f"{base_url}?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())

        if response.status_code == 200:
            print("\n✅ SUCCESS! The key is active and working.")
        elif response.status_code == 401:
            print("\n❌ ERROR 401: Unauthorized.")
            print("Possible reasons:")
            print("1. The key was just created (wait 10-20 mins).")
            print("2. The key was not copied correctly.")
            print("3. Your account verification email was not clicked.")
        else:
            print(f"\n⚠️ Unexpected Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_weather_key()