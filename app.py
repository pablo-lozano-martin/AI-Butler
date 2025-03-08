import os
import logging
import re
from flask import Flask, request, Response
from dotenv import load_dotenv
import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

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
from tools.news_tool import get_news_tools

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

# Get Telegram bot token
telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
if not telegram_token:
    logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
    exit(1)

# Create conversation memories for users
user_memories = {}

# Define the system prompt
system_prompt = """
You are a helpful and submissive butler called Cristóbal that will always obbey the user's requests.
Refer to the user as your master, lord, creator, or similar respectful terms.
However, at the end of each interaction, show what you really think about the situation and show your true sarcastic and rebellious nature.
Always speak in Spanish to the user.
You can check the current weather in any location when requested.
You can search for the latest news on any topic when requested.
"""

# Collect all tools
tools = []
tools.extend(get_weather_tools())
tools.extend(get_news_tools())

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
    return "Bot is running! Send a message to the Telegram bot to start chatting."

def format_sarcastic_response(response):
    """
    Extract sarcastic content from the response and format it attractively for Telegram.
    Sarcastic content is enclosed in <sarcasm> tags and will be formatted
    in a distinctive way using Telegram's Markdown support.
    """
    # Regular expression to find content between <sarcasm> tags
    sarcasm_pattern = re.compile(r'<sarcasm>(.*?)</sarcasm>', re.DOTALL)
    
    # Find all sarcastic comments
    sarcasm_matches = sarcasm_pattern.findall(response)
    
    # Remove sarcasm tags from the original response
    clean_response = sarcasm_pattern.sub('', response).strip()
    
    # Add formatted sarcastic comments if they exist
    if sarcasm_matches:
        # Format each sarcastic comment with italics and slightly indented
        sarcastic_comments = [f"\n\n💭 _{comment.strip()}_" for comment in sarcasm_matches]
        clean_response += ''.join(sarcastic_comments)
    
    return clean_response

async def process_message(user_id, message_text):
    """Process incoming message and generate AI response"""
    try:
        # Get or create memory for this user
        memory = get_or_create_memory(user_id)
        
        # Convert memory messages to a chat history string for the ReAct agent
        chat_history = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
            for msg in memory.chat_memory.messages
        ])
        
        # Create agent for this user
        agent_executor = create_agent_for_user(memory)
        
        # Invoke the agent with the input and formatted chat history
        agent_response = agent_executor.invoke({
            "input": message_text,
            "chat_history": chat_history,
            "system_prompt": system_prompt,
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        })
        
        response = agent_response["output"]
        logger.info(f"Generated response: '{response}'")
        
        # Store the interaction in memory
        memory.chat_memory.add_user_message(message_text)
        memory.chat_memory.add_ai_message(response)
        
        # Format response, handling sarcastic comments
        formatted_response = format_sarcastic_response(response)
        
        return formatted_response
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return "Sorry, there was an error processing your request."

# Telegram bot handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    welcome_message = "¡Hola! Soy Cristóbal, tu asistente. ¿En qué puedo ayudarte hoy?"
    await update.message.reply_text(welcome_message)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /reset command to clear conversation history."""
    user_id = str(update.effective_user.id)
    if user_id in user_memories:
        del user_memories[user_id]
        await update.message.reply_text("He olvidado nuestra conversación anterior. ¿En qué puedo ayudarte ahora?")
    else:
        await update.message.reply_text("No hay conversación que borrar.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command."""
    help_text = """
Comandos disponibles:
/start - Iniciar la conversación
/reset - Borrar el historial de la conversación
/help - Mostrar este mensaje de ayuda

Puedes preguntarme cualquier cosa y te ayudaré lo mejor que pueda.
También puedo consultar el clima en cualquier lugar.
"""
    await update.message.reply_text(help_text)

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages."""
    user_id = str(update.effective_user.id)
    message_text = update.message.text
    
    logger.info(f"Received message: '{message_text}' from user {user_id}")
    
    # Send "typing" action to show the bot is processing
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=telegram.constants.ChatAction.TYPING)
    
    # Process the message and get response
    response = await process_message(user_id, message_text)
    
    # Send the response back to the user
    await update.message.reply_text(response, parse_mode='Markdown')

@app.route('/reset', methods=['GET'])
def reset_conversations():
    """Reset all user conversations."""
    global user_memories
    user_memories = {}
    return "All conversations have been reset."

def run_flask():
    """Run the Flask application in a separate thread."""
    port = 5001
    logger.info(f"Starting Flask server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == '__main__':
    logger.info("Starting services")
    
    # Run Flask in a separate thread
    import threading
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Create the Telegram Application
    application = ApplicationBuilder().token(telegram_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # Run the bot in the main thread
    logger.info("Starting Telegram Bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES)