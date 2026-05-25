import streamlit as st
import requests
import time
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assamese ASR Demo",
    page_icon="🎙️",
    layout="centered",
)

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_GROUPS = {
    "🟠 Assamese Models": [
        ("Whisper Small – Assamese",                    "bijaykumarsingh/whisper-small-assamese"),
        ("Wav2Vec2 – Assamese",                         "bijaykumarsingh/wav2vec2-assamese"),
        ("IndicWav2Vec – Assamese (fine-tuned on BN)",  "bijaykumarsingh/indicwav2vec-assamese-asr-finetuned_on_bn"),
        ("IndicWav2Vec – Assamese ASR",                 "bijaykumarsingh/indicwav2vec-assamese-asr"),
        ("Wav2Vec2 – Assamese ASR",                     "bijaykumarsingh/wav2vec2-assamese-asr"),
        ("Whisper Turbo – Assamese (LDCIL)",            "bijaykumarsingh/whisper-turbo-as-LDCIL-sentence-Aligned_Dataset"),
        ("Whisper Small – Assamese CV17",               "bijaykumarsingh/whisper-small-as-cv17"),
    ],
    "🔵 Bengali Models": [
        ("Whisper Large v3 – Bengali CV17",  "bijaykumarsingh/whisper-large-v3-bn-cv17"),
        ("Whisper Turbo – Bengali CV17",     "bijaykumarsingh/whisper-turbo-bn-cv17"),
        ("Whisper Medium – Bengali CV17",    "bijaykumarsingh/whisper-medium-bn-cv17"),
        ("Whisper Small – Bengali CV17",     "bijaykumarsingh/whisper-small-bn-cv17"),
        ("Whisper Base – Bengali CV17",      "bijaykumarsingh/whisper-base-bn-cv17"),
        ("Whisper Tiny – Bengali CV17",      "bijaykumarsingh/whisper-tiny-bn-cv17"),
    ],
}

# Flat list for lookup: display_label → model_id
LABEL_TO_ID = {label: mid for group in MODEL_GROUPS.values() for label, mid in group}

# ── HF Inference helper ───────────────────────────────────────────────────────
MAX_RETRIES   = 12   # ~2 min wait for cold model
RETRY_DELAY   = 10   # seconds between retries

def call_inference_api(token: str, model_id: str, audio_bytes: bytes):
    """
    POST audio bytes to HF Inference API.
    Returns (text, None) on success or (None, error_message) on failure.
    Handles 503 model-loading retries automatically.
    """
    url     = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, data=audio_bytes, timeout=120)
        except requests.exceptions.Timeout:
            return None, "⏱️ Request timed out. The model may be overloaded – try again."
        except requests.exceptions.ConnectionError as e:
            return None, f"🔌 Connection error: {e}"

        if resp.status_code == 200:
            result = resp.json()
            # Whisper-style: {"text": "..."}
            if isinstance(result, dict) and "text" in result:
                return result["text"].strip(), None
            # Some models return a list
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"].strip(), None
            return str(result), None

        elif resp.status_code == 503:
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            est  = body.get("estimated_time", RETRY_DELAY)
            wait = min(float(est), RETRY_DELAY)
            st.toast(f"⏳ Model warming up… retry {attempt}/{MAX_RETRIES} (waiting {int(wait)}s)", icon="⏳")
            time.sleep(wait)

        elif resp.status_code == 401:
            return None, "🔑 Invalid or missing API token. Please check your token in the sidebar."

        elif resp.status_code == 404:
            return None, f"❌ Model `{model_id}` not found on HuggingFace."

        elif resp.status_code == 400:
            return None, f"⚠️ Bad request – the audio format may not be supported by this model.\n\n`{resp.text}`"

        else:
            return None, f"🚫 Unexpected error {resp.status_code}: {resp.text}"

    return None, "⌛ Model did not load in time. Please try again in a minute."

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
        width=40,
    )
    st.markdown("### 🔐 HuggingFace Token")
    hf_token = st.text_input(
        "Paste your HF API token",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Get your token at https://huggingface.co/settings/tokens",
    )
    st.caption("Your token is only used for this session and never stored.")

    st.divider()
    st.markdown("### 🤖 Select Model")

    selected_label = None
    for group_name, models in MODEL_GROUPS.items():
        st.markdown(f"**{group_name}**")
        for label, _ in models:
            if st.button(label, key=label, use_container_width=True):
                st.session_state["selected_model"] = label

    # Show active selection
    active = st.session_state.get("selected_model", list(LABEL_TO_ID.keys())[0])
    st.divider()
    st.success(f"**Active:** {active}")

    st.divider()
    st.markdown(
        "<small>Models run on HuggingFace cloud infrastructure. "
        "No audio is stored or shared beyond the API call.</small>",
        unsafe_allow_html=True,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎙️ Assamese Speech Recognition Demo")
st.markdown(
    "Upload an audio file and the selected model will transcribe it on the cloud. "
    "No model is downloaded to your device."
)

selected_label = st.session_state.get("selected_model", list(LABEL_TO_ID.keys())[0])
selected_id    = LABEL_TO_ID[selected_label]

st.info(f"**Model:** `{selected_id}`", icon="🧠")

# Audio uploader
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a", "webm"],
    help="Supported formats: WAV, MP3, OGG, FLAC, M4A, WEBM",
)

if uploaded_file:
    st.audio(uploaded_file, format=uploaded_file.type)
    audio_bytes = uploaded_file.read()

    transcribe_btn = st.button("▶ Transcribe", type="primary", use_container_width=True)

    if transcribe_btn:
        if not hf_token:
            st.error("🔑 Please enter your HuggingFace API token in the sidebar first.")
        else:
            with st.spinner("Sending audio to HuggingFace cloud… this may take a moment if the model is cold-starting."):
                text, err = call_inference_api(hf_token, selected_id, audio_bytes)

            if err:
                st.error(err)
            else:
                st.success("✅ Transcription complete!")
                st.markdown("### 📝 Transcription")
                st.markdown(
                    f"""<div style="
                        background:#f0f4ff;
                        border-left:4px solid #4f6ef7;
                        padding:16px 20px;
                        border-radius:8px;
                        font-size:1.15rem;
                        line-height:1.7;
                        color:#1a1a2e;
                    ">{text}</div>""",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label="⬇️ Download transcript (.txt)",
                    data=text,
                    file_name="transcription.txt",
                    mime="text/plain",
                )
else:
    st.markdown(
        """
        <div style="
            border: 2px dashed #ccc;
            border-radius:10px;
            padding: 40px;
            text-align:center;
            color:#888;
            margin-top:10px;
        ">
            📂 Upload an audio file above to get started
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built to showcase Assamese ASR models by "
    "[bijaykumarsingh](https://huggingface.co/bijaykumarsingh) · "
    "Powered by HuggingFace Inference API"
)
