import requests
import math

class WeatherDataFetcher:
    def get_weather_data(self):
        # Define API URL
        api_url = "https://api.open-meteo.com/v1/forecast?latitude=50.93&longitude=6.95&current=temperature_2m,windspeed_10m,winddirection_10m"
        # Request data
        response = requests.get(api_url)
        # Check if request and response is valid
        windspeed_10m = []
        winddirection_10m = []
        if response.status_code == 200:
            weather_data = response.json()
            windspeed_10m = weather_data['current']['windspeed_10m']
            winddirection_10m = weather_data['current']['winddirection_10m'] * math.pi / 180
        else:
            print("Failed to fetch weather data.")

        return windspeed_10m, winddirection_10m

    def save_weather_data(self):
        # Hole die Wetterdaten
        windspeed, winddirection = self.get_weather_data()

        # Speichere die Wetterdaten in den Instanzvariablen
        self.saved_windspeed = windspeed
        self.saved_winddirection = winddirection