import os
import streamlit as st
from langchain_community.agent_toolkits import GmailToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import find_dotenv, load_dotenv
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from email.mime.text import MIMEText
import base64
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import datetime
import pytz
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dateutil import parser as dateutil_parser
import dateparser
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import shutil
import subprocess
import sqlite3
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.utilities import SerpAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper


# Load environment variables

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")




# Check required API keys
if not all([OPENAI_API_KEY, SERPAPI_API_KEY, GOOGLE_API_KEY, CSE_ID,LANGSMITH_API_KEY]):
    st.error("Some API keys are missing. Please check your .env file.")

# Set up SQLite connection for Personal Information
conn = sqlite3.connect("personal_info.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS personal_info (
        key TEXT PRIMARY KEY,
        value TEXT
    )
""")
conn.commit()

# Helper function to store and retrieve info from SQLite
def store_info(key, value):
    cursor.execute("INSERT OR REPLACE INTO personal_info (key, value) VALUES (?, ?)", (key, value))
    conn.commit()

def get_info(key):
    cursor.execute("SELECT value FROM personal_info WHERE key = ?", (key,))
    result = cursor.fetchone()
    return result[0] if result else None

# Helper function to query LLM using Ollama
def query_llama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        return f"Exception occurred: {str(e)}"

# Streamlit UI setup
st.set_page_config(page_title="Multi-Tool App", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an application:", ["Email Sender", "Meeting Scheduler", "PDF Summarizer", "Internet Search", "Personal Information Manager"])

# Email Sender function
def email_sender():
    st.title("Email Sender")
    # Activate API keys

  # Gmail Authentication and Setup
    def authenticate_gmail():
        credentials = get_gmail_credentials(
            token_file="token_email.json",
            scopes=["https://mail.google.com/"],
            client_secrets_file="credentials_email.json",
        )
        api_resource = build_resource_service(credentials=credentials)
        toolkit = GmailToolkit(api_resource=api_resource)
        return toolkit.get_tools(), build('gmail', 'v1', credentials=credentials)

    tools, gmail_service = authenticate_gmail()

    # Set up the LangChain Agent with instructions to generate a subject and body
    instructions = """
    You are an assistant that creates email drafts with a subject and body.
    Please respond in the following format:
    Subject: <generated subject>
    Body: <generated email body>

    The email should:
    1. Start with a greeting that includes "Hi" followed by the recipient's first name.
    2. End with "Best regards, Harshavardhana" as the closing.
    """

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

    def load_contacts():
        with open("contacts.json", "r") as f:
            return json.load(f)

    contacts = load_contacts()

    # Helper function to process chat and get subject and body
    def process_chat(agent_executor, user_input, chat_history):
        # Extract recipient's first name from email
        recipient_name = st.session_state["recipient_name"].split('(')[0].title()
        
        # Add instructions to the input to specify greeting and closing
        prompt_with_name = (
            f"{user_input}\n\n"
            f"Please start the email with 'Hi {recipient_name},' and end with 'Best regards, Harshavardhana.'"
        )
        
        response = agent_executor.invoke({
            "input": prompt_with_name,
            "chat_history": chat_history
        })
        full_output = response["output"]

        # Parse the subject and body from the response with a fallback mechanism
        subject, body = "No Subject Generated", "No Content Generated"
        if "Subject:" in full_output and "Body:" in full_output:
            try:
                subject = full_output.split("Subject: ")[1].split("Body: ")[0].strip()
                body = full_output.split("Body: ")[1].strip()
            except IndexError:
                st.warning("Could not parse the subject and body correctly.")

        # Safeguard for empty intermediate steps
        intermediate_output = response.get('intermediate_steps', [])
        intermediate_message = intermediate_output[0].tool_input['message'] if intermediate_output else "No Intermediate Output"

        return subject, body, intermediate_message

    # Streamlit UI Layout
    st.title("Email Drafting App with LangChain")
    st.write("Use this app to draft and send emails via Gmail.")

    # Initialize session state for chat history and draft
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "draft_body" not in st.session_state:
        st.session_state["draft_body"] = ""
    if "draft_subject" not in st.session_state:
        st.session_state["draft_subject"] = ""
    if "recipient_email" not in st.session_state:
        st.session_state["recipient_email"] = None


    # Select recipient from dropdown
    recipient_selection = st.selectbox("Select a recipient", options=list(contacts.keys()))
    st.session_state["recipient_email"] = contacts[recipient_selection]
    st.session_state['recipient_name']=recipient_selection
    # Enter email prompt
    user_input = st.text_area("Enter your email prompt", height=150)

    # Step 1: Check Draft Button
    if st.button("Check Draft"):
        # Process chat to generate subject and draft body
        chat_history = st.session_state["chat_history"]
        generated_subject, generated_body, intermediate_message = process_chat(agent_executor, user_input, chat_history)
        
        # Store subject and body in session state
        st.session_state["draft_subject"] = generated_subject
        st.session_state["draft_body"] = generated_body

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=f"Subject: {generated_subject}\n\nBody: {generated_body}"))
        st.session_state["chat_history"] = chat_history

        # Display the generated subject and draft for review
        st.write("### Generated Email Subject:")
        st.write(generated_subject)
        st.write("### Generated Email Draft:")
        st.write(generated_body)

        # Display intermediate tool output (optional)
        #st.write("### Intermediate Tool Output:")
        #st.write(intermediate_message)

    # Step 2: Send Email Button
    if st.button("Send Email"):
        recipient_email = st.session_state.get("recipient_email")
        email_subject = st.session_state.get("draft_subject")
        email_body = st.session_state.get("draft_body")

        # Ensure recipient email, subject, and draft body are set properly
        if not recipient_email or not email_subject.strip() or not email_body.strip():
            st.error("The email draft or recipient email is missing. Please generate a valid draft before sending.")
        else:
            # Send the email using Gmail API
            try:
                message = MIMEText(email_body)
                message['to'] = recipient_email
                message['subject'] = email_subject

                raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
                message_body = {'raw': raw_message}

                # Send the email
                send_message = gmail_service.users().messages().send(userId="me", body=message_body).execute()
                st.success(f"Email successfully sent to {recipient_email}.")
            except HttpError as error:
                st.error(f"An error occurred: {error}")


# Meeting Scheduler function
def meeting_scheduler():
    st.title("Meeting Scheduler")

    # Initialize OpenAI Chat Model
    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Google Calendar Authentication
    def authenticate_google():
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        creds = None

        if os.path.exists('token_meeting.json'):
            creds = Credentials.from_authorized_user_file('token_meeting.json', SCOPES)
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials_meeting.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token_meeting.json', 'w') as token:
                token.write(creds.to_json())
        return creds

    def get_calendar_service():
        creds = authenticate_google()
        service = build('calendar', 'v3', credentials=creds)
        return service

    # Retrieve free slots for a specific day
    def get_free_slots_for_day(date):
        service = get_calendar_service()
        tz = pytz.timezone('America/Chicago')

        start_of_day = tz.localize(datetime.datetime.combine(date, datetime.time.min))
        end_of_day = tz.localize(datetime.datetime.combine(date, datetime.time.max))

        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_of_day.isoformat(),
            timeMax=end_of_day.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        booked_slots = [
            (event['start'].get('dateTime'), event['end'].get('dateTime'))
            for event in events
        ]

        free_slots = []
        for hour in range(9, 18):  # Business hours from 9:00 to 18:00
            slot_start = tz.localize(datetime.datetime.combine(date, datetime.time(hour, 0)))
            slot_end = slot_start + datetime.timedelta(hours=1)

            if all(
                not (slot_start <= datetime.datetime.fromisoformat(start) < slot_end)
                for start, end in booked_slots
            ):
                free_slots.append(f"{slot_start.strftime('%H:%M')} - {slot_end.strftime('%H:%M')}")

        return free_slots

    # LLM-powered command parsing function
    def parse_llm_command(command):
        tz = pytz.timezone('America/Chicago')

        today = datetime.datetime.now(tz).date()

        prompt = f"""
        Extract the meeting name, start time, end time, and date from the following input:

        Input: "{command}"

        If the input mentions 'today', use '{today}' as the date.
        If the input uses a relative date (like 'in two days' or 'next Monday',etc can be of any type like 'next tuesday','next week','next month','next tuesday')etc.,,, handle it appropriately.

        Provide the response in the following JSON format:
        {{
            "meeting_name": "<name>",
            "start_time": "<HH:MM>",
            "end_time": "<HH:MM>",
            "date": "<YYYY-MM-DD>"
        }}
        """

        response = chat([HumanMessage(content=prompt)])

        try:
            parsed_data = json.loads(response.content.strip())
            return parsed_data
        except Exception as e:
            st.error(f"Error parsing command with LLM: {e}")
            return None

    # Schedule a meeting in Google Calendar
    def schedule_meeting(meeting_name, start_time, end_time):
        service = get_calendar_service()
        event = {
            'summary': meeting_name,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': 'America/Chicago'
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': 'America/Chicago'
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }

        # Check for conflicts before scheduling
        if not is_time_available(start_time, end_time):
            st.error(f"Conflict detected with another meeting. Please choose a different time.")
            return

        event_result = service.events().insert(calendarId='primary', body=event).execute()
        return f"Meeting '{meeting_name}' scheduled successfully from {start_time} to {end_time}."

    # Check if a time slot is available
    def is_time_available(start_time, end_time):
        service = get_calendar_service()
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        return len(events) == 0

    # Streamlit UI Layout
    st.title("Interactive Google Calendar Meeting Scheduler")

    # Find Free Slots Section
    # Find Free Slots Section
    st.header("Find Free Slots")
    date_input = st.date_input("Select a date to find free slots:")

    if st.button("Search Free Slots"):
        try:
            parsed_date = date_input  # Already a datetime.date object from st.date_input
            free_slots = get_free_slots_for_day(parsed_date)

            if free_slots:
                # Display the date in 'Oct 21, 2024' format
                formatted_date = parsed_date.strftime('%b %d, %Y')
                st.write(f"Free slots on {formatted_date}:")
                st.write(" | ".join(free_slots))
            else:
                st.write(f"No free slots available on {formatted_date}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    # Schedule a Meeting via Command Section
    st.header("Schedule a Meeting via Command")
    llm_command = st.text_input(
        "Type your command (e.g., 'Schedule the meeting Team Sync between 13:30 and 14:30 on Oct 21, 2024'): "
    )

    if st.button("Execute Command"):
        parsed_data = parse_llm_command(llm_command)

        if parsed_data:
            try:
                # Extract meeting details from the parsed JSON
                meeting_name = parsed_data["meeting_name"]
                start_time_str = parsed_data["start_time"]
                end_time_str = parsed_data["end_time"]
                date = datetime.datetime.strptime(parsed_data["date"], "%Y-%m-%d").date()

                # Schedule the meeting using the extracted details
                tz = pytz.timezone('America/Chicago')
                start_time = tz.localize(datetime.datetime.combine(date, datetime.datetime.strptime(start_time_str, "%H:%M").time()))
                end_time = tz.localize(datetime.datetime.combine(date, datetime.datetime.strptime(end_time_str, "%H:%M").time()))

                response = schedule_meeting(meeting_name, start_time, end_time)
                st.success(response)
            except Exception as e:
                st.error(f"Error scheduling meeting: {e}")
        else:
            st.error("Invalid command format. Please try again.")


# PDF Summarizer function
def pdf_summarizer():
    st.title("PDF Summarizer")

    def get_pdf_text(pdf_docs):
        all_text = ""
        total_docs = len(pdf_docs)
        progress_bar = st.progress(0)

        for idx, pdf in enumerate(pdf_docs):
            pdf_text = ""
            try:
                st.info(f"Processing {pdf.name} ({idx + 1}/{total_docs})...")
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {e}. Skipping this file.")
                continue

            if pdf_text.strip():
                all_text += pdf_text + "\n\n"
                st.success(f"{pdf.name} processed successfully.")
            else:
                st.warning(f"No content found in {pdf.name}. Skipping.")

            progress_bar.progress((idx + 1) / total_docs)

        if not all_text.strip():
            st.warning("No text extracted from the uploaded PDFs.")
        return all_text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def create_faiss_index(text_chunks):
        try:
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
        except PermissionError as e:
            st.error(f"PermissionError: {e}. Close any program accessing 'faiss_index' and try again.")
            return

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in the context, respond with 
        "answer is not available in the context." Do not provide incorrect answers.\n\n
        Context:\n{context}\n
        Question:\n{question}\n
        Answer:
        """
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def user_input(user_question):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)

            if not docs:
                st.warning("No relevant documents found.")
                return

            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response.get("output_text", "No response generated."))
        except Exception as e:
            st.error(f"An error occurred: {e}")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question and st.session_state.get("processed", False):
        user_input(user_question)

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Submit & Process") and pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            if raw_text.strip():
                text_chunks = get_text_chunks(raw_text)
                create_faiss_index(text_chunks)
                st.session_state["processed"] = True
                st.success("Processing complete! You can now ask questions.")

    if st.session_state.get("processed", False):
        st.info("PDFs are processed. You can now ask questions.")

# Internet Search function
def internet_search():
    st.title("Internet Search")

    # Initialize APIs
    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    google_search = GoogleSearchAPIWrapper()

    def get_dynamic_search_agent():
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

        agent = initialize_agent(
            tools,
            chat,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent

    def run_user_query(query):
        agent = get_dynamic_search_agent()
        try:
            response = agent.run(query)
            st.write("Response:", response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    user_question = st.text_input("Ask a Question and let the agent search for you")
    if user_question.strip():
        with st.spinner("Searching the web..."):
            run_user_query(user_question)

# Personal Information Manager function
def personal_info_manager():
    st.title("Personal Information Manager")
    st.header("Store Personal Information")
    
    # Input fields for personal information
    name = st.text_input("Enter your name")
    dob = st.date_input("Enter your date of birth")
    height = st.number_input("Enter your height (in cm)", min_value=0)
    weight = st.number_input("Enter your weight (in kg)", min_value=0)
    hobby = st.text_input("Enter your hobby")
    color = st.text_input("Enter your favorite color")
    food = st.text_input("Enter your favorite food")

    # Store information in SQLite
    if st.button("Store Information"):
        store_info("name", name)
        store_info("date_of_birth", dob.strftime('%Y-%m-%d'))
        store_info("height", str(height))
        store_info("weight", str(weight))
        store_info("hobby", hobby)
        store_info("color", color)
        store_info("food", food)
        st.success("All information stored successfully!")

    # Retrieve and display stored information
    st.header("Retrieve Personal Information")
    if st.button("Show My Information"):
        retrieved_data = {
            "Name": get_info("name"),
            "Date of Birth": get_info("date_of_birth"),
            "Height (cm)": get_info("height"),
            "Weight (kg)": get_info("weight"),
            "Favorite Color": get_info("color"),
            "Hobby": get_info("hobby"),
            "Favorite Food": get_info("food")
        }
        for key, value in retrieved_data.items():
            if value:
                st.write(f"**{key}:** {value}")

    # Query LLM with personal data
    st.header("Ask the LLM a Question")
    llm_prompt = st.text_area("Ask the LLM a question using your personal data")
    if st.button("Ask LLM"):
        prompt = f"My name is {get_info('name')}. I was born on {get_info('date_of_birth')}. " \
                 f"My favorite color is {get_info('color')}, and I love {get_info('food')}. " \
                 f"I am {get_info('height')} cm tall and weigh {get_info('weight')} kg. " \
                 f"In my free time, I enjoy {get_info('hobby')}. {llm_prompt}"
        response = query_llama(prompt)
        st.write(f"LLM Response: {response}")

# Render the selected application
if app_mode == "Email Sender":
    email_sender()
elif app_mode == "Meeting Scheduler":
    meeting_scheduler()
elif app_mode == "PDF Summarizer":
    pdf_summarizer()
elif app_mode == "Internet Search":
    internet_search()
elif app_mode == "Personal Information Manager":
    personal_info_manager()