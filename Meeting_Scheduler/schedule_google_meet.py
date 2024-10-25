import os
import datetime
import pytz  # Add this to handle time zones
import streamlit as st
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

# Authenticate with Google Calendar API
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

# Build Google Calendar service
def get_calendar_service():
    creds = authenticate_google()
    service = build('calendar', 'v3', credentials=creds)
    return service

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

# Get available slots on a given day
def get_available_slots(date):
    tz = pytz.timezone('America/Chicago')  # Set to CST time zone
    available_slots = []

    for hour in range(9, 18):  # Business hours (9:00 AM to 6:00 PM)
        start_time = tz.localize(datetime.datetime.combine(date, datetime.time(hour, 0)))
        end_time = start_time + datetime.timedelta(hours=1)

        if is_time_available(start_time, end_time):
            available_slots.append(f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")

    return available_slots

# Schedule a meeting in Google Calendar
def schedule_meeting(event_name, start_time, end_time):
    service = get_calendar_service()
    event = {
        'summary': event_name,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/Chicago'  # Ensure correct time zone
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'America/Chicago'  # Ensure correct time zone
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }

    event_result = service.events().insert(calendarId='primary', body=event).execute()
    return f"Meeting '{event_name}' scheduled successfully from {start_time} to {end_time}."

# Streamlit UI Layout
st.title("Google Calendar Meeting Scheduler")

# Initialize session state
if "available_slots" not in st.session_state:
    st.session_state.available_slots = None

if "selected_slot" not in st.session_state:
    st.session_state.selected_slot = None

# User selects the meeting date
meeting_date = st.date_input("Select the meeting date")

# Check available slots when the user clicks the button
if st.button("Check Available Slots"):
    st.session_state.available_slots = get_available_slots(meeting_date)

# Display available slots if they exist
if st.session_state.available_slots:
    selected_slot = st.selectbox("Available Slots", st.session_state.available_slots)

    # Store the selected slot in session state
    st.session_state.selected_slot = selected_slot

    # User inputs the meeting name
    event_name = st.text_input("Enter the meeting name")

    if st.button("Schedule Meeting"):
        slot_start_str, slot_end_str = st.session_state.selected_slot.split(" - ")
        tz = pytz.timezone('America/Chicago')  # Time zone for CST

        # Parse and localize the start and end times
        slot_start = tz.localize(datetime.datetime.combine(
            meeting_date, datetime.datetime.strptime(slot_start_str, "%H:%M").time()
        ))
        slot_end = tz.localize(datetime.datetime.combine(
            meeting_date, datetime.datetime.strptime(slot_end_str, "%H:%M").time()
        ))

        response = schedule_meeting(event_name, slot_start, slot_end)
        st.success(response)
