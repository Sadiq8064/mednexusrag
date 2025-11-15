# Use Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "rag:app", "--host", "0.0.0.0", "--port", "8000"]
