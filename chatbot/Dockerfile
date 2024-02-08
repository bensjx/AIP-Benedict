# Use an official Python runtime as a parent image
FROM python:3.10.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install the Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run
CMD ["python",  "app.py"]