FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

COPY . .

RUN mkdir -p data

EXPOSE 8000

CMD ["python", "run.py"]
