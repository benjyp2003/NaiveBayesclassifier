# Use a Python image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app
ENV PYTHONPATH=/app

# Copy the requirements file
COPY building_app/requirements.txt .

# Add this section to install system dependencies
RUN apt-get update && apt-get install -y gcc libevdev-dev linux-headers-generic

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY building_app /app/building_app

# Expose the port the building_app runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["python", "-u", "-m", "uvicorn", "building_app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]