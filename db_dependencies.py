import subprocess
import sqlite3
from datetime import datetime, timedelta
import os

def install_libraries():
    required_libraries = [
        "streamlit", "transformers", "torch", "nltk", "matplotlib", "plotly",
        "pandas", "numpy", "langdetect"
    ]
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            print(f"Installing {lib}...")
            subprocess.check_call(["pip", "install", lib])

def populate_feedback_database():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()

    # Create the feedback table if it doesn't exist (only text and date)
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            text TEXT NOT NULL,
            date TEXT NOT NULL
        )
    ''')

    # Sample feedback entries (only text and date)
    sample_entries = [
        {
            "text": "The delivery was incredibly fast and the quality was amazing! However, one of the clothing items didn't fit well.",
            "date": datetime.now() - timedelta(days=5)
        },
        {
            "text": "Great service, fast delivery, but the material quality is poor.",
            "date": datetime.now() - timedelta(days=3)
        },
        {
            "text": "Love the clothes, perfect fit, but delivery took too long.",
            "date": datetime.now() - timedelta(days=1)
        },
        {
            "text": "The product quality is excellent, but the delivery was delayed.",
            "date": datetime.now() - timedelta(days=2)
        },
        {
            "text": "Fast delivery and stylish clothes, but the fit was too tight.",
            "date": datetime.now() - timedelta(days=4)
        }
    ]

    # Insert sample entries into the database
    for entry in sample_entries:
        c.execute('INSERT INTO feedback VALUES (?, ?)', (entry["text"], entry["date"].isoformat()))

    conn.commit()
    conn.close()
    print("Sample feedback entries added to feedback.db")

def run_streamlit_app():
    if not os.path.exists("run.py"):
        raise FileNotFoundError("run.py not found in the current directory. Please save the Streamlit dashboard code as survey.py.")

    install_libraries()
    populate_feedback_database()

    try:
        subprocess.run(["streamlit", "run", "run.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except KeyboardInterrupt:
        print("Streamlit app stopped by user.")

if __name__ == "__main__":
    try:
        run_streamlit_app()
    except Exception as e:
        print(f"An error occurred: {e}")
