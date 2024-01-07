# Use an official Python runtime as a parent image
FROM python:3.9-slim

# The environment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /Server/models
WORKDIR /Server/models

# Copy the requirements file into the container at /Server/models
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /Server/models
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run backend.py when the container launches
CMD ["python", "./backend.py"]
