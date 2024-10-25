import subprocess
import sqlite3
import streamlit as st

# Initialize SQLite connection and create a table for personal information
conn = sqlite3.connect("personal_info.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS personal_info (
        key TEXT PRIMARY KEY,
        value TEXT
    )
""")
conn.commit()

# Function to store information in SQLite
def store_info(key, value):
    cursor.execute("INSERT OR REPLACE INTO personal_info (key, value) VALUES (?, ?)", (key, value))
    conn.commit()

# Function to retrieve information from SQLite
def get_info(key):
    cursor.execute("SELECT value FROM personal_info WHERE key = ?", (key,))
    result = cursor.fetchone()
    return result[0] if result else None

# Updated function to query the local LLM using Ollama
def query_llama(prompt):
    try:
        # Use subprocess to run Ollama and capture output
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=prompt,  # Pass the prompt directly as a string
            stdout=subprocess.PIPE,         # Capture standard output
            stderr=subprocess.PIPE,         # Capture standard error
            text=True                        # Ensure output is returned as text
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        return f"Exception occurred: {str(e)}"

    
# Streamlit UI layout
st.title("Personal Information Manager with SQLite and Ollama")

# Section to store multiple personal attributes
st.header("Store Personal Information")
name = st.text_input("Enter your name")
dob = st.date_input("Enter your date of birth")
height = st.number_input("Enter your height (in cm)", min_value=0)
weight = st.number_input("Enter your weight (in kg)", min_value=0)
hobby=st.text_input("Enter your hobby")
color=st.text_input("Enter your favorite color")
food=st.text_input("Enter your favorite food")


if st.button("Store Information"):
    # Store all the information in SQLite
    store_info("name", name)
    store_info("date_of_birth", dob.strftime('%Y-%m-%d'))
    store_info("height", str(height))
    store_info("weight", str(weight))
    store_info("hobby", hobby)
    store_info("color", color)
    store_info("food",food)

    st.success("All information stored successfully!")

# Section to retrieve and display personal information
st.header("Retrieve Personal Information")
if st.button("Show My Information"):
    retrieved_data = {
        "Name": get_info("name"),
        "Date of Birth": get_info("date_of_birth"),
        "Height (cm)": get_info("height"),
        "Weight (kg)": get_info("weight"),
        "Color":get_info("color"),
        "Hobby":get_info("hobby"),
        "Food":get_info("food")
    }
    for key, value in retrieved_data.items():
        if value:
            st.write(f"**{key}:** {value}")

# Section to query the LLM using stored data
st.header("Ask the LLM a Question")
llm_prompt = st.text_area("Ask the LLM a question using your personal data")




if st.button("Ask LLM"):
    # Construct a prompt using the stored information
    prompt = f"My name is {get_info('name')}. I was born on {get_info('date_of_birth')}. " \
             f"My favorite color is {get_info('favorite_color')}, and I love {get_info('favorite_food')}. " \
             f"I am {get_info('height')} cm tall and weigh {get_info('weight')} kg. " \
             f"In my free time, I enjoy {get_info('hobby')}. {llm_prompt}"

    # Query the LLM and display the response
    response = query_llama(prompt)
    st.write(f"LLM Response: {response}")

