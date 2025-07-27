import os
import time
from app.api.client import Client
from app.controller.manager import Manager

def run_automated_client_task():
    """This function contains the client logic that will be triggered automatically on server startup."""
    # Give the server a moment to be fully available
    time.sleep(3)
    print("--- 🚀 Triggering automated client task from main.py ---")
    try:
        manager = Manager()
        manager.start()
        print("--- ✅ Automated client task finished ---")
    except Exception as e:
        print(f"--- ❌ Error during automated client task: {e} ---")

if __name__ == '__main__':
    run_automated_client_task()
