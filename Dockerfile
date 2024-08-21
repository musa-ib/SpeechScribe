# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and the Streamlit app to the container
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY SpeechCom.keras /app/SpeechCom.keras

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
