FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3-pip \
    curl \
    ca-certificates \
    git

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY camembert.py /app/camembert.py
COPY train.csv /app/train.csv
COPY test.csv /app/test.csv

CMD ["python3", "camembert.py"]
