
import subprocess

def run_streamlit_app():
    """Runs the Streamlit app as a subprocess."""
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except FileNotFoundError:
        print("Error: Streamlit command not found. Make sure Streamlit is installed in your environment.")

if __name__ == "__main__":
    print("Starting the main application...")
    run_streamlit_app()
    print("Main application finished.")