from classifying_app.api.client import Client


class Manager:

    def __init__(self):
        """Initialize the Manager with necessary configurations."""
        self.client = Client()