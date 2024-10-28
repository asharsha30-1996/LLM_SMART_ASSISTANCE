import os
import subprocess

# Define the base directory where projects are located
BASE_DIR = os.path.abspath("C:\\Users\\Hp\\Desktop\\LLM_SMART_ASSISTANCE")  # Adjusted base directory

# Map each project to its Streamlit entry script
projects = {
    "EMAIL_SENDER": os.path.join("EMAIL_SENDER", "email_sender.py"),
    "Meeting_Scheduler": os.path.join("Meeting_Scheduler", "schedule_google_meet.py"),
    "PDF_READER_SUMMARIZER": os.path.join("PDF_READER_SUMMARIZER", "PDF_SUMMARIZER.py"),
    "REAL_TIME_INTERNET_SEARCH": os.path.join("REAL_TIME_INTERNET_SEARCH", "internet_search.py"),
    "Personal_Info": os.path.join("Personal_Info", "ollama_personal_retrieval.py"),
}

def list_projects():
    """Display all available projects."""
    print("Available Projects:")
    for i, project_name in enumerate(projects, start=1):
        print(f"{i}. {project_name}")

def get_project_selection():
    """Prompt the user to select a project."""
    list_projects()
    try:
        choice = int(input("\nEnter the project number to run: "))
        if 1 <= choice <= len(projects):
            return list(projects.keys())[choice - 1]
        else:
            print("Invalid choice. Please try again.")
            return get_project_selection()
    except ValueError:
        print("Invalid input. Please enter a number.")
        return get_project_selection()

def run_project(project_name):
    """Run the selected project using Streamlit."""
    script_path = projects[project_name]
    project_dir = os.path.dirname(script_path)  # Extract the project directory

    # Change the working directory to the project folder
    os.chdir(os.path.join(BASE_DIR, project_dir))

    print(f"\nRunning {project_name} from {os.getcwd()}...")

    try:
        # Run the Streamlit app using the relative path
        subprocess.run(["streamlit", "run", os.path.basename(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {project_name}: {e}")
    finally:
        # Switch back to the base directory after execution
        os.chdir(BASE_DIR)

if __name__ == "__main__":
    # Get the user's project selection
    selected_project = get_project_selection()
    
    # Run the selected project
    run_project(selected_project)
