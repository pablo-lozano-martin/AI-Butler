import logging
import requests
import os
from dotenv import load_dotenv
from langchain.tools import StructuredTool

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get API key from environment
OPEN_WEATHER_API_KEY = os.environ.get('OPEN_WEATHER_API_KEY')

def get_coordinates(location: str):
    """Convert location name to coordinates using OpenWeatherMap Geocoding API"""
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPEN_WEATHER_API_KEY}"
        response = requests.get(geo_url)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            logger.warning(f"No coordinates found for location: {location}")
            return None, None
            
        lat = data[0]['lat']
        lon = data[0]['lon']
        return lat, lon
    except Exception as e:
        logger.error(f"Error getting coordinates for {location}: {str(e)}")
        return None, None

def get_weather(location: str) -> str:
    """Get the current weather in a given location"""
    logger.info(f"Weather requested for: {location}")
    
    if not OPEN_WEATHER_API_KEY:
        logger.error("OpenWeatherMap API key not found in environment variables")
        return "Sorry, the weather service is not available at the moment. Confifure the API key and try again."
    
    try:
        # Get coordinates for the location
        lat, lon = get_coordinates(location)
        
        if not lat or not lon:
            return f"Could not find the location: {location}"
        
        # Make API call to OpenWeatherMap
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPEN_WEATHER_API_KEY}&units=metric&lang=es"
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant weather information
        city_name = data.get('name', location)
        country = data.get('sys', {}).get('country', '')
        
        weather_main = data.get('weather', [{}])[0].get('main', '')
        weather_desc = data.get('weather', [{}])[0].get('description', '')
        
        temp = data.get('main', {}).get('temp')
        feels_like = data.get('main', {}).get('feels_like')
        humidity = data.get('main', {}).get('humidity')
        
        wind_speed = data.get('wind', {}).get('speed')
        
        # Format response in Spanish
        weather_info = f"Weather in {city_name}"
        if country:
            weather_info += f", {country}"
        
        weather_info += f":\n• Condition: {weather_desc.capitalize()}\n"
        
        if temp is not None:
            weather_info += f"• Temperature: {temp}°C\n"
        if feels_like is not None:
            weather_info += f"• Feels like: {feels_like}°C\n"
        if humidity is not None:
            weather_info += f"• Humidity: {humidity}%\n"
        if wind_speed is not None:
            weather_info += f"• Wind speed: {wind_speed} m/s"
        
        logger.info(f"Weather data retrieved for {location}: {weather_info}")
        return weather_info
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error getting weather for {location}: {str(e)}")
        return f"Error getting the weather at {location}. Could not connect to the weather service."
    except Exception as e:
        logger.error(f"Error getting weather for {location}: {str(e)}")
        return f"Error getting the weather at {location}."

# Create a structured tool for the weather function
weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="get_weather",
    description="Useful for when you need to get the current weather in a specific location."
)

# Function to get all tools from this module
def get_weather_tools():
    """Return all weather-related tools"""
    return [weather_tool]