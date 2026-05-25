import streamlit as st
import requests
import time
from audio_recorder_streamlit import audio_recorder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assamese ASR Demo",
    page_icon="🎙️",
    layout="centered",
)

# ── HF API – NEW router endpoint (api-inference.huggingface.co is deprecated) ─
HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_GROUPS = {
    "🟠 Assamese Models": [
        ("Whisper Small – Assamese",                   "bijaykumarsingh/whisper-small-assamese"),
        ("Wav2Vec2 – Assamese",                        "bijaykumarsingh/wav2vec2-assamese"),
        ("IndicWav2Vec – Assamese (fine-tuned on BN)", "bijaykumarsingh/indicwav2vec-assamese-asr-finetuned_on_bn"),
        ("IndicWav2Vec – Assamese ASR",                "bijaykumarsingh/indicwav2vec-assamese-asr"),
        ("Wav2Vec2 – Assamese ASR",                    "bijaykumarsingh/wav2vec2-assamese-asr"),
        ("Whisper Turbo – Assamese (LDCIL)",           "bijaykumarsingh/whisper-turbo-as-LDCIL-sentence-Aligned_Dataset"),
        ("Whisper Small – Assamese CV17",              "bijaykumarsingh/whisper-small-as-cv17"),
    ],
    "🔵 Bengali Models": [
        ("Whisper Large v3 – Bengali CV17", "bijaykumarsingh/whisper-large-v3-bn-cv17"),
        ("Whisper Turbo – Bengali CV17",    "bijaykumarsingh/whisper-turbo-bn-cv17"),
        ("Whisper Medium – Bengali CV17",   "bijaykumarsingh/whisper-medium-bn-cv17"),
        ("Whisper Small – Bengali CV17",    "bijaykumarsingh/whisper-small-bn-cv17"),
        ("Whisper Base – Bengali CV17",     "bijaykumarsingh/whisper-base-bn-cv17"),
        ("Whisper Tiny – Bengali CV17",     "bijaykumarsingh/whisper-tiny-bn-cv17"),
    ],
}

LABEL_TO_ID = {label: mid for group in MODEL_GROUPS.values() for label, mid in group}

# ── Inference helper ──────────────────────────────────────────────────────────
MAX_RETRIES = 12
RETRY_DELAY = 10

def call_inference_api(token: str, model_id: str, audio_bytes: bytes):
    """
    POST audio bytes to HuggingFace router (new endpoint, replaces
    the deprecated api-inference.huggingface.co).
    Returns (text, None) on success or (None, error_str) on failure.
    """
    url     = f"{HF_API_BASE}/{model_id}"
    headers = {
        "Authorization":  f"Bearer {token}",
        "Content-Type":   "audio/wav",
        "X-Wait-For-Model": "true",   # ask router to wait instead of 503-ing immediately
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, data=audio_bytes, timeout=120)
        except requests.exceptions.Timeout:
            return None, "⏱️ Request timed out. Try again or use a shorter clip."
        except requests.exceptions.ConnectionError as e:
            return None, (
                "🔌 Cannot reach HuggingFace.\n\n"
                "Check your internet connection and try again.\n\n"
                f"`{e}`"
            )

        if resp.status_code == 200:
            try:
                result = resp.json()
            except Exception:
                return None, f"⚠️ Unexpected non-JSON response: {resp.text[:300]}"

            if isinstance(result, dict) and "text" in result:
                return result["text"].strip(), None
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"].strip(), None
            return str(result), None

        elif resp.status_code == 503:
            try:
                wait = min(float(resp.json().get("estimated_time", RETRY_DELAY)), RETRY_DELAY)
            except Exception:
                wait = RETRY_DELAY
            st.toast(f"⏳ Model warming up… retry {attempt}/{MAX_RETRIES} ({int(wait)}s)", icon="⏳")
            time.sleep(wait)

        elif resp.status_code == 401:
            return None, "🔑 Invalid API token. Check the token in the sidebar."
        elif resp.status_code == 404:
            return None, (
                f"❌ Model `{model_id}` not found or not supported by the Inference API.\n\n"
                "Make sure the model is public and has an enabled Inference endpoint on HuggingFace."
            )
        elif resp.status_code == 400:
            return None, (
                f"⚠️ Bad request – this model may not support the audio format you uploaded.\n\n"
                f"`{resp.text[:300]}`"
            )
        elif resp.status_code == 422:
            return None, f"⚠️ Unprocessable audio – try converting to WAV 16 kHz mono.\n\n`{resp.text[:300]}`"
        else:
            return None, f"🚫 Error {resp.status_code}: {resp.text[:300]}"

    return None, "⌛ Model did not load in time. Please try again in a minute."


def show_transcription_result(token, model_id, audio_bytes):
    """Run inference and render result in the UI."""
    with st.spinner("Sending to HuggingFace cloud… (model may need a moment to warm up)"):
        text, err = call_inference_api(token, model_id, audio_bytes)

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
                line-height:1.8;
                color:#1a1a2e;
            ">{text}</div>""",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=text,
            file_name="transcription.txt",
            mime="text/plain",
        )


# ── Session state ─────────────────────────────────────────────────────────────
if "selected_model"      not in st.session_state:
    st.session_state["selected_model"] = list(LABEL_TO_ID.keys())[0]
if "last_recorded_bytes" not in st.session_state:
    st.session_state["last_recorded_bytes"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=36)
    st.markdown("### 🔐 HuggingFace Token")
    hf_token = st.text_input(
        "Paste your HF API token",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Get your token at https://huggingface.co/settings/tokens  (Read access is enough)",
    )
    st.caption("Used only for this session. Never stored.")

    st.divider()
    st.markdown("### 🤖 Select Model")

    for group_name, models in MODEL_GROUPS.items():
        st.markdown(f"**{group_name}**")
        for label, _ in models:
            is_active = st.session_state["selected_model"] == label
            if st.button(
                label,
                key=f"btn_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["selected_model"] = label
                st.rerun()

    st.divider()
    st.caption("All inference runs on HuggingFace cloud. No audio is stored.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎙️ Assamese Speech Recognition")
st.markdown(
    "Record live or upload an audio file — transcription happens entirely on the cloud. "
    "Nothing is downloaded to your device."
)

active_label = st.session_state["selected_model"]
active_id    = LABEL_TO_ID[active_label]
st.info(f"**Active model:** `{active_id}`", icon="🧠")
st.divider()

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_record, tab_upload = st.tabs(["🎤  Record Audio", "📂  Upload File"])

# ─── Tab 1: Live recording ────────────────────────────────────────────────────
with tab_record:
    st.markdown(
        "Press 🔴 to **start** recording. Press again to **stop** — "
        "transcription starts automatically."
    )
    _, col_mic, _ = st.columns([1, 2, 1])
    with col_mic:
        recorded_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#4f6ef7",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=3.0,
            key="mic_recorder",
        )

    if recorded_bytes and recorded_bytes != st.session_state["last_recorded_bytes"]:
        st.session_state["last_recorded_bytes"] = recorded_bytes
        st.audio(recorded_bytes, format="audio/wav")
        if not hf_token:
            st.warning("🔑 Enter your HuggingFace API token in the sidebar to transcribe.")
        else:
            show_transcription_result(hf_token, active_id, recorded_bytes)

# ─── Tab 2: File upload ───────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a", "webm"],
        help="Best results with WAV 16 kHz mono. Use ffmpeg to convert: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav",
    )
    if uploaded:
        st.audio(uploaded, format=uploaded.type)
        upload_bytes = uploaded.read()
        if st.button("▶ Transcribe", type="primary", use_container_width=True, key="transcribe_btn"):
            if not hf_token:
                st.error("🔑 Enter your HuggingFace API token in the sidebar first.")
            else:
                show_transcription_result(hf_token, active_id, upload_bytes)
    else:
        st.markdown(
            """<div style="border:2px dashed #ccc;border-radius:10px;
               padding:36px;text-align:center;color:#999;">
               📂 Upload an audio file to get started
            </div>""",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built to showcase Assamese ASR models by "
    "[bijaykumarsingh](https://huggingface.co/bijaykumarsingh) · "
    "Powered by [HuggingFace Inference API](https://huggingface.co/inference-api)"
)
