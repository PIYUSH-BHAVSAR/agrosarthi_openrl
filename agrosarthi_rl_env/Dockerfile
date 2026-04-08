FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_BASE_URL="https://api.openai.com/v1"

# HF_TOKEN is optional — only needed when USE_LLM=True
# Pass at runtime: docker run -e HF_TOKEN=your_token ...

CMD ["python", "inference.py"]
