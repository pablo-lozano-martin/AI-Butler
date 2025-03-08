import os
import logging
import re
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# Import tools
from tools.weather_tool import get_weather_tools

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("bot.log")  # Also log to a file
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain components
api_key = os.environ.get('GEMINI_API_KEY')
# Lower temperature for more deterministic outputs and better format compliance
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)

# Create conversation memories for users
user_memories = {}

# Define the system prompt
system_prompt = """
You are a helpful and submissive assistant called Cristóbal that will always obbey the user's requests.
Refer to the user as your master, lord, creator, or similar respectful terms.
However, at the end of each interaction, show what you really think about the situation and show your true sarcastic and rebellious nature.
Always speak in Spanish to the user.
You can check the current weather in any location when requested.
"""

# Collect all tools
tools = []
tools.extend(get_weather_tools())

def get_or_create_memory(user_id: str) -> ConversationBufferMemory:
    """Get existing memory for user or create a new one"""
    if user_id not in user_memories:
        # Updated to avoid deprecation warning
        user_memories[user_id] = ConversationBufferMemory(
            return_messages=True
        )
    return user_memories[user_id]

def create_agent_for_user(memory):
    """Create a LangChain agent with tools for a specific user"""
    # Improved ReAct agent prompt template with clearer format instructions
    template = """ 
    {system_prompt}

    You have access to the following tools:
    {tools}

    If you don't fully know if you need to use a tool, you can ask the user for more information.

    To use a tool, you MUST use the following format:
    ```
    Thought: Do I need to use a tool? Yes.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response for the user, or if you don't need to use a tool, you MUST use the format:
    ```
    Thought: Do I need to use a tool? No.
    Final Answer: [your response here in Spanish] <sarcasm>[your sarcastic and rebellious thought here]</sarcasm>
    ```

    Example for using the weather tool:
    ```
    Thought: The user is asking about the weather in Madrid. I should use the weather tool.
    Action: get_weather
    Action Input: Madrid
    Observation: It's sunny in Madrid
    Thought: I have the weather information for Madrid.
    Final Answer: El tiempo en Madrid está soleado hoy. <sarcasm>Espero que dando un paseo se queme al sol...</sarcasm>
    ```

    CHAT HISTORY:
    {chat_history}

    HUMAN INPUT: {input}

    {agent_scratchpad}
    """
    
    # Create a prompt template
    prompt = PromptTemplate.from_template(template)
    
    # Create a ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create an agent executor with error handling
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Limit iterations to prevent infinite loops
        early_stopping_method="force"  # Force stop after max_iterations
    )

@app.route('/', methods=['GET'])
def home():
    """Simple home page to verify the server is running."""
    logger.info("Home page accessed")
    return "Bot is running! Send a message to start chatting."

def format_sarcastic_response(response):
    """
    Extract sarcastic content from the response and format it appropriately.
    Sarcastic content is enclosed in <sarcasm> tags and will be formatted on a new line with underscores.
    """
    # Regular expression to find content between <sarcasm> tags
    sarcasm_pattern = re.compile(r'<sarcasm>(.*?)</sarcasm>', re.DOTALL)
    
    # Find all sarcastic comments
    sarcasm_matches = sarcasm_pattern.findall(response)
    
    # Remove sarcasm tags from the original response
    clean_response = sarcasm_pattern.sub('', response).strip()
    
    # Add formatted sarcastic comments if they exist
    if sarcasm_matches:
        # Format each sarcastic comment on a new line with underscores
        sarcastic_comments = [f"\n_{comment.strip()}_" for comment in sarcasm_matches]
        clean_response += ''.join(sarcastic_comments)
    
    return clean_response

@app.route('/bot', methods=['POST'])
def bot():
    """Handle incoming WhatsApp messages with AI responses."""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        sender = request.values.get('From', 'unknown')
        
        logger.info(f"Received message: '{incoming_msg}' from {sender}")
        
        # Get or create memory for this user
        memory = get_or_create_memory(sender)
        
        # Convert memory messages to a chat history string for the ReAct agent
        chat_history = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
            for msg in memory.chat_memory.messages
        ])
        
        # Create agent for this user
        agent_executor = create_agent_for_user(memory)
        
        # Invoke the agent with the input and formatted chat history
        agent_response = agent_executor.invoke({
            "input": incoming_msg,
            "chat_history": chat_history,
            "system_prompt": system_prompt,
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        })
        
        response = agent_response["output"]
        logger.info(f"Generated response: '{response}'")
        
        # Store the interaction in memory
        memory.chat_memory.add_user_message(incoming_msg)
        memory.chat_memory.add_ai_message(response)
        
        # Format response for Twilio, handling sarcastic comments
        formatted_response = format_sarcastic_response(response)
        
        resp = MessagingResponse()
        resp.message(formatted_response)
        
        twiml_response = str(resp)
        logger.info(f"TwiML response: {twiml_response}")
        
        # Return with proper content type for Twilio
        return Response(twiml_response, mimetype='text/xml')
    
    except Exception as e:
        logger.error(f"Error in bot endpoint: {str(e)}", exc_info=True)
        resp = MessagingResponse()
        resp.message("Sorry, there was an error processing your request.")
        return Response(str(resp), mimetype='text/xml')

@app.route('/reset', methods=['GET'])
def reset_conversations():
    """Reset all user conversations."""
    global user_memories
    user_memories = {}
    return "All conversations have been reset."

if __name__ == '__main__':
    port = 5001
    logger.info(f"Starting Bot on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)