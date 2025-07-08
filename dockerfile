# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy project files
COPY . .

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
