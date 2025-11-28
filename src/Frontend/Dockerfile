# base image
FROM python:3.11-slim

# setting a working directory
WORKDIR /app

#copying requirements and installing dependeces
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501


# CMD ["streamlit","run","main.py"] -> for local 
CMD ["streamlit","run","main.py","--server.address=0.0.0.0","--server.port=8501"]