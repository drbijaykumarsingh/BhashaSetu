# 🎙️ Assamese ASR Demo — Streamlit App

A lightweight demo app for transcribing Assamese (and Bengali) speech using
models hosted on HuggingFace. All inference runs on HuggingFace cloud — nothing
is downloaded to the user's device.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Run locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo and set the main file to `app.py`.
4. (Optional) Add your HF token as a **secret** so it pre-fills:
   - In Streamlit Cloud dashboard → **Secrets** → add:
     ```toml
     HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
     ```
   - Then in `app.py`, replace the text_input default with:
     ```python
     import os
     hf_token = st.text_input(..., value=os.environ.get("HF_TOKEN", ""))
     ```
5. Click **Deploy**.

---

## 🤗 Getting a HuggingFace API Token

1. Log in at https://huggingface.co
2. Go to **Settings → Access Tokens**
3. Create a token with **Read** permissions
4. Paste it into the app sidebar

---

## 📋 Models included

| Model | Language |
|---|---|
| whisper-small-assamese | Assamese |
| wav2vec2-assamese | Assamese |
| indicwav2vec-assamese-asr-finetuned_on_bn | Assamese |
| indicwav2vec-assamese-asr | Assamese |
| wav2vec2-assamese-asr | Assamese |
| whisper-turbo-as-LDCIL-sentence-Aligned_Dataset | Assamese |
| whisper-small-as-cv17 | Assamese |
| whisper-large-v3-bn-cv17 | Bengali |
| whisper-turbo-bn-cv17 | Bengali |
| whisper-medium-bn-cv17 | Bengali |
| whisper-small-bn-cv17 | Bengali |
| whisper-base-bn-cv17 | Bengali |
| whisper-tiny-bn-cv17 | Bengali |

---

## 🎵 Supported Audio Formats

WAV · MP3 · OGG · FLAC · M4A · WEBM

> **Tip:** WAV (16 kHz, mono) gives the most reliable results with Wav2Vec2 models.
