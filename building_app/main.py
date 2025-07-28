import os
import time
from building_app.api.client import Client
from building_app.controller.manager import Manager

def run_automated_client_task():
    """This function contains the client logic that will be triggered automatically on server startup."""
    time.sleep(3)
    print("\n----  Triggering automated client task from main.py  ----")
    try:
        manager = Manager()
        manager.start_process()

        print("----  Automated client task finished  ----")
    except Exception as e:
        print(f"----  Error during automated client task: {e}  ----")

if __name__ == '__main__':
    run_automated_client_task()
