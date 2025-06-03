# Dockerfile
FROM python:3.10

# Define working directory
WORKDIR '/app'

# Copy all dir
COPY . .

# Install any python packages you need
COPY /requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
