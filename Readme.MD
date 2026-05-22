# QuizCraft AI — RAG-Powered Quiz Maker

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

Or edit `app.py` line:
```python
client = OpenAI(api_key="your-openai-api-key")
```

## Run
```bash
python app.py
```
Open http://localhost:5000

## How it works

1. **Extract** — PDF text is extracted from your selected page range
2. **Chunk** — Text is split into 500-word overlapping chunks
3. **Embed** — Each chunk is embedded using OpenAI `text-embedding-3-small`
4. **Index** — Embeddings stored in a FAISS vector index
5. **Retrieve** — For each question, relevant chunks are retrieved via similarity search
6. **Generate** — GPT-3.5-Turbo generates a mixed question from retrieved context

## Question Types
- Multiple Choice (MCQ) — 4 options, 1 correct
- True / False
- Fill in the Blank — key term replaced with _____
- Short Answer — specific word/phrase from text

## Project Structure
```
quiz_app/
├── app.py
├── requirements.txt
└── templates/
    ├── index.html
    ├── quiz.html
    └── result.html
```