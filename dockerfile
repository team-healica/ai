FROM ubuntu:22.04

ENV PYTHON_VERSION=3.11

RUN apt-get update -y && \
    apt-get install -y git && \
    apt-get install -y python${PYTHON_VERSION} && \
    apt-get install -y python3-pip && \
    apt-get clean

WORKDIR /app

RUN git clone https://github.com/team-healica/ai.git

WORKDIR /app/ai

RUN pip3 install -r requirements.txt

EXPOSE 3000

CMD ["uvicorn", "server:app"]