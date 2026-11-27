FROM python:3.14-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD python run.py
