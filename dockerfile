# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python packages
# We add --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
# This includes the 'core' and 'db' folders, etc.
COPY . .

# This line runs the ingestion script when the image is built.
# It ensures the ChromaDB and BM25 indexes are pre-built into the container.
RUN python ingest.py

# Expose the port the API will run on
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]