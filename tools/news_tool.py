import logging
import requests
import os
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from typing import Optional

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get API key from environment
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

def get_news(query: str, category: Optional[str] = None, country: Optional[str] = None) -> str:
    """
    Get the latest news based on search query, category, and/or country.
    
    Args:
        query (str): The search term to look for in news articles
        category (str, optional): News category (business, entertainment, general, health, science, sports, technology)
        country (str, optional): The 2-letter ISO 3166-1 code of the country (e.g., 'us', 'mx', 'es')
    
    Returns:
        str: A formatted string with the latest news
    """
    logger.info(f"News requested for: query={query}, category={category}, country={country}")
    
    if not NEWS_API_KEY:
        logger.error("NewsAPI key not found in environment variables")
        return "Sorry, the news service is not available at the moment. Configure the API key and try again."
    
    try:
        # Base URL for everything endpoint
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": query,
            "pageSize": 5,  # Limit to 5 articles
            "language": "es"  # Preferably Spanish results
        }
        
        # If category and country are specified, use the top-headlines endpoint instead
        if category or country:
            url = "https://newsapi.org/v2/top-headlines"
            if category:
                params["category"] = category
            if country:
                params["country"] = country
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check if we got any results
        if data.get('totalResults', 0) == 0:
            return f"No se encontraron noticias para '{query}'"
        
        # Format the news response
        news_results = ["Últimas noticias:"]
        
        for idx, article in enumerate(data.get('articles', []), 1):
            title = article.get('title', 'Sin título')
            source = article.get('source', {}).get('name', 'Fuente desconocida')
            description = article.get('description', 'Sin descripción')
            url = article.get('url', '')
            
            news_item = f"{idx}. {title} ({source})\n   {description}\n   {url}\n"
            news_results.append(news_item)
        
        return "\n".join(news_results)
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error getting news: {str(e)}")
        return f"Error obteniendo noticias. No se pudo conectar al servicio de noticias."
    except Exception as e:
        logger.error(f"Error getting news: {str(e)}")
        return f"Error obteniendo noticias: {str(e)}"

# Create a structured tool for the news function
news_tool = StructuredTool.from_function(
    func=get_news,
    name="get_news",
    description="Useful for getting the latest news on a specific topic, category, or from a specific country. Provide a query string and optionally a category and/or country code."
)

# Function to get all news tools from this module
def get_news_tools():
    """Return all news-related tools"""
    return [news_tool]
