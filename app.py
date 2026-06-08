"""
Plant Disease Detection System - Root Wrapper
This script automatically runs the Streamlit web dashboard located in the app/ directory.
"""
import sys
import subprocess

if __name__ == "__main__":
    print("Launching Streamlit dashboard from the structured app/ directory...")
    try:
        subprocess.run(["streamlit", "run", "app/app.py"] + sys.argv[1:])
    except KeyboardInterrupt:
        print("\nStopping dashboard service.")
