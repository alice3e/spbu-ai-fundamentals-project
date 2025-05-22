# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY ./app/ .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Command to run the application
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
CMD ["flask", "run", "--host=0.0..0", "--port=5000"]
# For development, you might use: CMD ["flask", "run", "--host=0.0..0", "--port=5000"]