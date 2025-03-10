# 🤖 AI Butler (Cristóbal)

## 📝 Description

Cristóbal is a sarcastic yet helpful AI butler that runs as a Telegram bot. This assistant combines the power of Gemini AI with useful tools to provide information and assistance while maintaining a unique personality.

## ✨ Features

- 💬 Natural language conversation powered by Gemini 2.0
- 🌤️ Real-time weather information for any location
- 📰 Latest news search functionality
- 💾 Conversation memory that persists between interactions
- 😏 Unique personality: formal and helpful on the surface with sarcastic undertones

## 🛠️ Technical Overview

The project uses:
- **Flask**: Lightweight web server
- **Python Telegram Bot**: Interface with Telegram API
- **LangChain**: Framework for composing AI applications
- **Gemini AI**: Google's large language model
- **Custom tools**: Weather and news search capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Telegram Bot Token
- Google Gemini API Key

### Environment Variables

Create a `.env` file in the root directory with:

GEMINI_API_KEY=your_gemini_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPEN_WEATHER_API_KEY=your_openweather_api_key
NEWS_API_KEY=your_newsapi_key

## 🔧 Tools

### Weather Tool

Fetches real-time weather data from OpenWeatherMap API for any location. The tool provides:

- Current temperature
- Weather conditions
- Humidity
- Wind speed
- "Feels like" temperature

## 💡 Usage

Start a conversation with your Telegram bot

Use commands:

- /start - Begin interaction
- /help - List available commands
- /reset - Clear conversation history

Ask for weather: "¿Cuál es el clima en Barcelona hoy?"

Request news: "Muéstrame noticias sobre tecnología en España"

Ask general questions or engage in conversation

## 🔮 Future Enhancements

- Integration with calendar services
- Translation capabilities
- Image generation support
- Voice message handling
- More advanced conversation memory
