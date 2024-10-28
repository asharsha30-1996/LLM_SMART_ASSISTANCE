import os
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
 

# Validate API keys
if not OPENAI_API_KEY or not SERPAPI_API_KEY or not GOOGLE_API_KEY:
    st.error("API keys not found. Please check your .env file.")
    st.stop()

# Initialize APIs
chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
google_search = GoogleSearchAPIWrapper()

def get_dynamic_search_agent():
    """Create a LangChain agent with multiple search tools."""
    tools = [
        Tool(
            name="Google CSE Search",
            func=google_search.run,
            description="Use this tool to search the web using Google Custom Search Engine."
        ),
        Tool(
            name="SerpAPI",
            func=serp_search.run,
            description="Use this tool for real-time and detailed search results."
        )
    ]

    # Initialize the agent
    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

def run_user_query(query):
    """Handle user queries."""
    agent = get_dynamic_search_agent()
    try:
        response = agent.run(query)
        st.write("Response:", response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Dynamic Search Agent", page_icon="🌐", layout="wide")
    st.header("Dynamic Multi-API Search Agent 🌐")

    user_question = st.text_input("Ask a Question and let the agent search for you")

    if user_question.strip():
        with st.spinner("Searching the web..."):
            run_user_query(user_question)

    with st.sidebar:
        st.title("About")
        st.write(
            """
            This agent uses multiple APIs and GPT-based reasoning to decide what to search 
            and where to search. Ask any question and get the latest information from the web!
            """
        )

if __name__ == "__main__":
    main()
