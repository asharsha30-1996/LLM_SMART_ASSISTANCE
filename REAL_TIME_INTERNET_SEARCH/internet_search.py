import os
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Configure OpenAI API
if not OPENAI_API_KEY or not SERPAPI_API_KEY:
    st.error("API keys not found. Please check your .env file.")
    st.stop()

# Initialize the chat model using OpenAI GPT
chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Initialize SerpAPI search tool
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

def get_search_agent():
    """Create a LangChain agent with Google Search capabilities."""
    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="Use this tool to search the web and get real-time information."
        )
    ]

    # Initialize the agent with tools and OpenAI model
    agent = initialize_agent(
        tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return agent

def user_query(query):
    """Handle user queries and provide responses from the agent."""
    agent = get_search_agent()
    try:
        # Generate response from the agent
        response = agent.run(query)
        st.write("Response:", response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Real-time Internet Search Agent", page_icon="🌐", layout="wide")
    st.header("Real-time Internet Search Agent 🌐")

    user_question = st.text_input("Ask a Question and get real-time information from the web")

    if user_question:
        with st.spinner("Searching the web..."):
            user_query(user_question)

    with st.sidebar:
        st.title("About")
        st.write(
            """
            This agent searches the internet using SerpAPI and responds in real-time. 
            Ask any question to get the latest web-based information.
            """
        )

if __name__ == "__main__":
    main()
