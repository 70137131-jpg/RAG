# How to Run the RAG Project

## Prerequisites

1. **Install Dependencies** (already done):
   ```powershell
   pip install -r requirements.txt
   ```

2. **Set up API Key**:
   Create a `.env` file in the project root with one of these:
   ```
   GOOGLE_API_KEY=your-google-key-here
   ```
   OR
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
   OR
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```
   OR
   ```
   DEEPSEEK_API_KEY=sk-your-key-here
   ```

## Running the Project

### Option 1: Quick Start Demo (Recommended for first time)
Runs a simple demo with 50 samples:
```powershell
python quickstart.py
```

### Option 2: Data Ingestion (ingest.py)
Load data into the vector store. This must be run before using the web app:

**Basic usage:**
```powershell
python ingest.py
```

**With options:**
```powershell
# Use fast configuration
python ingest.py --config fast

# Use balanced configuration (default)
python ingest.py --config balanced

# Use accurate configuration
python ingest.py --config accurate

# Limit number of samples
python ingest.py --samples 50

# Custom collection name
python ingest.py --collection my_collection
```

### Option 3: Web Application (app.py)
Run the Flask web interface:

```powershell
python app.py
```

Then open your browser to: `http://localhost:5000`

**Note:** You must run `ingest.py` first to populate the vector store before the web app will work.

## Typical Workflow

1. **First time setup:**
   ```powershell
   # 1. Create .env file with your API key
   # 2. Ingest data
   python ingest.py --config balanced --samples 100
   
   # 3. Run the web app
   python app.py
   ```

2. **Quick test:**
   ```powershell
   python quickstart.py
   ```

## Configuration Profiles

- **fast**: Smaller model, faster responses (gemini-1.5-flash)
- **balanced**: Good balance (default, gemini-1.5-flash)
- **accurate**: Larger model, better accuracy (gemini-1.5-flash)
- **gemini**: Gemini 1.5 Flash configuration

## Troubleshooting

- **"API key not found"**: Make sure you created a `.env` file with your API key
- **"Vector store is empty"**: Run `python ingest.py` first
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
