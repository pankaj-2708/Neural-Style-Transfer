# base image
FROM python:3.11-slim

# setting a working directory
WORKDIR /app

#copying requirements and installing dependeces
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cv2 dep
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*


COPY . .

EXPOSE 8000

CMD ["python","app.py"]