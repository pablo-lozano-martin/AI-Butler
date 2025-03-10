import logging
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.tools import StructuredTool
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def search_internet(query: str, num_results: int = 3) -> str:
    """
    Search the internet using DuckDuckGo and return results.
    
    Args:
        query (str): The search query
        num_results (int, optional): Number of results to return. Defaults to 3.
    
    Returns:
        str: A formatted string with search results
    """
    logger.info(f"Internet search requested for: query={query}, num_results={num_results}")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        if not results:
            return f"No se encontraron resultados para la búsqueda: '{query}'"
        
        # Format the search results
        formatted_results = [f"Resultados de búsqueda para '{query}':"]
        
        for idx, result in enumerate(results, 1):
            title = result.get('title', 'Sin título')
            body = result.get('body', 'No hay descripción disponible')
            href = result.get('href', '')
            
            formatted_results.append(f"{idx}. {title}\n   {body}\n   {href}\n")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error searching the internet: {str(e)}")
        return f"Error realizando la búsqueda en Internet: {str(e)}"

def get_webpage_content(url: str, max_length: int = 1000) -> str:
    """
    Get the text content of a webpage.
    
    Args:
        url (str): URL of the webpage to extract content from
        max_length (int, optional): Maximum length of content to return. Defaults to 1000.
    
    Returns:
        str: The extracted text content of the webpage
    """
    logger.info(f"Webpage content requested for: {url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator="\n")
        
        # Clean up text - break into lines and remove leading/trailing spaces
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate if necessary
        if len(text) > max_length:
            text = text[:max_length] + "... (contenido truncado)"
        
        return f"Contenido de {url}:\n\n{text}"
        
    except Exception as e:
        logger.error(f"Error getting webpage content: {str(e)}")
        return f"Error obteniendo el contenido de la página web: {str(e)}"

# Create structured tools
internet_search_tool = StructuredTool.from_function(
    func=search_internet,
    name="search_internet",
    description="Useful for searching the internet to find information on any topic. Provide a query string and optionally the number of results to return."
)

webpage_content_tool = StructuredTool.from_function(
    func=get_webpage_content,
    name="get_webpage_content",
    description="Useful for fetching the content of a specific webpage. Provide the URL of the webpage and optionally the maximum length of content to return."
)

# Function to get all internet tools from this module
def get_internet_tools():
    """Return all internet-related tools"""
    return [internet_search_tool, webpage_content_tool]