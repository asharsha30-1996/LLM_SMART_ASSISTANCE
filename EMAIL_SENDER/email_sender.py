import os
import streamlit as st
from langchain_community.agent_toolkits import GmailToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import find_dotenv, load_dotenv
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

# Activate API keys
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Gmail Authentication and Setup
def authenticate_gmail():
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    return toolkit.get_tools()

tools = authenticate_gmail()

# Set up the LangChain Agent
instructions = """You are an assistant that creates email drafts."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

# Helper function to process chat
def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return [response["output"], response['intermediate_steps'][0]]

# Streamlit UI Layout
st.title("Email Drafting App with LangChain")
st.write("Use this app to draft emails and interact with Gmail via LangChain.")

# Initialize session state to maintain chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_area("Enter your email prompt", height=150)

if st.button("Submit"):
    # Load chat history from session state
    chat_history = st.session_state["chat_history"]

    # Process the chat with the agent executor
    response = process_chat(agent_executor, user_input, chat_history)

    # Update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response[0]))

    # Store updated chat history in session state
    st.session_state["chat_history"] = chat_history

    # Display the generated email draft
    st.write("### Generated Email Draft:")
    st.write(response[0])

    # Display intermediate tool output (optional)
    st.write("### Intermediate Tool Output:")
    st.write(response[1][0].tool_input['message'])

