FROM python:3.11-slim

WORKDIR /app

# install deps first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app files
COPY catalog.json .
COPY catalog.py .
COPY search.py .
COPY agent.py .
COPY main.py .

# pre-download the sentence-transformers model at build time
# so there's no cold-start delay waiting for model download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
