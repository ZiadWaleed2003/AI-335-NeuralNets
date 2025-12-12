import subprocess
import sys
import os


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gui_path = os.path.join(script_dir, "src", "GUI.py")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", gui_path,
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
