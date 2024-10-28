import os
import json
import datetime
import pytz
import streamlit as st
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dateutil import parser as dateutil_parser
import dateparser

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Chat Model
chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Google Calendar Authentication
def authenticate_google():
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
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
    If the input uses a relative date (like 'in two days' or 'next Monday'), handle it appropriately.

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
