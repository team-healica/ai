FROM ubuntu:22.04

ENV PYTHON_VERSION=3.11

RUN apt-get update -y && \
    apt-get install -y git && \
    apt-get install -y python${PYTHON_VERSION} && \
    apt-get install -y python3-pip && \
    apt-get clean

RUN mkdir /app/

WORKDIR /app/

RUN git clone https://github.com/team-healica/ai.git

WORKDIR /app/ai

RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN apt-get update -y && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0

CMD ["uvicorn", "server:app", "--port", "5000"]