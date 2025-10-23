# AI Mental Health Support Bot (Streamlit + Hugging Face)

A beginner-friendly, sentiment-aware support chatbot that uses a transformer model to tailor empathetic responses. It is not medical advice and not a substitute for professional care.

- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- UI: Streamlit
- NLP: Hugging Face `transformers`

---

## 1) Prerequisites
- Python 3.10+ (Windows)
- Internet connection (first run downloads ~300MB of model files)
- PowerShell (recommended)

Optional:
- Git (for version control)

---

## 2) Setup (Windows, step-by-step)
Run these in PowerShell in the project folder `mental health chatbot/`.

```powershell
# 2.1 Create a virtual environment
python -m venv .venv

# 2.2 Activate the venv
.\.venv\Scripts\Activate.ps1

# (If activation is blocked, temporarily allow scripts only for this session)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2.3 Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Run the app
```powershell
streamlit run app.py
```
The app will open in your browser. If it doesn’t, copy the `Local URL` shown in the terminal into your browser.

---

## 4) How it works
- Uses a Hugging Face sentiment pipeline to classify your message as POSITIVE/NEGATIVE/NEUTRAL with a confidence score.
- Simple rule-based logic turns sentiment + certain keywords into empathetic responses.
- A lightweight crisis-language check offers helpline resources if concerning terms are detected. Always call your local emergency number in an emergency (e.g., 112/911/999).

---

## 5) Folder structure
```
mental health chatbot/
├─ app.py                # Streamlit app
├─ requirements.txt      # Python dependencies
└─ README.md             # Setup and usage
```

---

## 6) Next steps (learn & extend)
- Swap models: try multilingual or larger models in `app.py` inside `load_sentiment_pipeline()`.
- Add memory: store past messages or user mood over time.
- Add quick actions: breathing timer, grounding exercises, journaling prompts.
- Improve safety: add a more thorough keyword list, region-aware helplines.

---

## 7) Optional: Switch to Flask UI later
If you prefer Flask over Streamlit:
- Create `app_flask.py` with a basic Flask server and a `/analyze` endpoint that calls the same Hugging Face pipeline.
- Use a simple HTML/JS frontend (or a template engine) to send user input via `fetch` to `/analyze` and render the response.
- You can reuse the response logic from `build_empathetic_response()`.

---

## 8) Safety and disclaimer
- This chatbot provides supportive conversation, not diagnosis or treatment.
- In a crisis or emergency, call your local emergency number or visit the nearest emergency department.
- International helplines directory: https://findahelpline.com/
