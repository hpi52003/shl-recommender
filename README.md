# SHL Assessment Recommender

A conversational agent that helps hiring managers find the right SHL assessments through dialogue.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file
cp .env.example .env
# Open .env and paste your Groq API key
# Get a free key at: https://console.groq.com

# 3. Run locally
uvicorn main:app --reload --port 8000
```

## Get your Groq API key (free)

1. Go to https://console.groq.com
2. Sign up with email or Google
3. Click "API Keys" in the left sidebar
4. Click "Create API Key" — copy the key (starts with gsk_...)
5. Paste it in your .env file:
   GROQ_API_KEY=gsk_...your_key_here

## Test it

```bash
# Health check
curl http://localhost:8000/health

# Chat example
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer, mid-level"}
    ]
  }'
```

## Run the trace tests

```bash
python test_traces.py
```

## Deploy to Render (free)

1. Push to GitHub
2. Go to render.com → New Web Service → connect your repo
3. Set environment variable: GROQ_API_KEY=your_key  (NOT in code, in Render dashboard)
4. Build command: pip install -r requirements.txt
5. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

## API

### GET /health
Returns {"status": "ok"} with HTTP 200.

### POST /chat
Request:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

Response:
```json
{
  "reply": "Here are 3 assessments that fit your needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

recommendations is [] when still gathering context.
end_of_conversation is true only when user confirms the list is final.

## Project structure

```
shl_recommender/
  catalog.json      # SHL product catalog (377 assessments)
  catalog.py        # Loads and parses the catalog
  search.py         # FAISS semantic search
  agent.py          # LLM prompt + response validation (uses Groq)
  main.py           # FastAPI app
  test_traces.py    # Tests against sample conversation patterns
  requirements.txt
  Dockerfile
  .env.example      # Template — copy to .env and add your key
  .gitignore        # Keeps .env out of git
```
