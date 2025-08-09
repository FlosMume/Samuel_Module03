### import libraries in venv

# numpy has been installed in venv
import numpy
print("Numpy version:", numpy.__version__)

# torch for Cuda has been installed in venn
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
print("CUDA version (used by PyTorch):", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available. Check installation.")

# Verify GPU usage
x = torch.tensor([1.0, 2.0, 3.0]).cuda() # Create a tensor on the GPU
y = x * 2 # Perform an operation
print(y)  # Should show the result on the GPU
print(y.device)  # Should output: cuda:0

### FastAPI Skeleton for my Voice Assistant
# Class3HomeworkSamuel.py
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Dict, List

from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env into environment

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

import uvicorn


# ASR
from faster_whisper import WhisperModel

# LLM (OpenAI optional)
import openai  # Always import
from openai import OpenAI
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test the API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello, who are you?"}
    ]
)

print(response.choices[0].message.content)

# Transformers fallback
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# TTS
import edge_tts

# Config
BASE_DIR = Path(__file__).parent.resolve()
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)
RESPONSE_AUDIO = BASE_DIR / "response.wav"

app = FastAPI(title="LLM Voice Assistant - Week3")

# --- Simple 5-turn memory store (in-memory) ---
# Map session_id -> list of dicts [{"role":"user"/"assistant","text":...}, ...]
CONV_MEMORY: Dict[str, List[Dict[str, str]]] = {}
MAX_TURNS = 5  # keep last 5 exchanges (user+assistant count as 1 turn here)

# --- Models: initialize lazily to avoid long startup ---
_asr_model = None
_llm_pipeline = None
_local_gen_model = None


def get_asr_model(model_name: str = "small", device: str = "cuda"):
    global _asr_model
    if _asr_model is None:
        # model_name examples: "small", "medium", "large-v2", or a huggingface path
        _asr_model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "float32")
    return _asr_model


def get_local_generator(model_name: str = "gpt2", device_map="auto"):
    global _llm_pipeline, _local_gen_model
    if OPENAI_API_KEY:
        return None  # will use OpenAI
    if _llm_pipeline is None:
        # WARNING: gpt2 is tiny, helpful for tests. Replace with a chat model you have downloaded for real use.
        # If you have a larger model on disk, set model_name accordingly.
        try:
            _llm_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_length=512,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        except Exception as e:
            print("Local generator pipeline init failed:", e)
            _llm_pipeline = None
    return _llm_pipeline


# --- Utilities ---
def save_upload_file_tmp(upload_file: UploadFile, tmp_dir: Path) -> Path:
    suffix = Path(upload_file.filename).suffix or ".wav"
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return tmp_path


def add_to_memory(session_id: str, role: str, text: str):
    conv = CONV_MEMORY.setdefault(session_id, [])
    conv.append({"role": role, "text": text})
    # enforce max turns: we consider 2 messages = 1 turn roughly; keep last MAX_TURNS*2 messages
    max_msgs = MAX_TURNS * 2
    if len(conv) > max_msgs:
        CONV_MEMORY[session_id] = conv[-max_msgs:]


def build_prompt_from_memory(session_id: str, new_user_text: str) -> str:
    conv = CONV_MEMORY.get(session_id, [])
    # Build a chat-style prompt. For simple models, produce a single concatenated prompt.
    prompt_parts = []
    for msg in conv:
        if msg["role"] == "user":
            prompt_parts.append(f"User: {msg['text']}")
        else:
            prompt_parts.append(f"Assistant: {msg['text']}")
    prompt_parts.append(f"User: {new_user_text}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


# --- Step 1: ASR with faster-whisper ---
async def run_asr(audio_path: Path, model_name: str = "small", device: str = "cuda") -> str:
    """
    Transcribe audio file using faster-whisper.
    Returns transcribed text.
    """
    model = await asyncio.to_thread(get_asr_model, model_name, device)
    # model.transcribe is blocking; run in thread
    segments, info = await asyncio.to_thread(model.transcribe, str(audio_path))
    # segments is an iterator/list of Segment objects with .text
    text = "".join([seg.text for seg in segments])
    return text.strip()


# --- Step 2: Generate response with LLM ---
async def generate_response(prompt: str, max_tokens: int = 200) -> str:
    """
    Prefer OpenAI if OPENAI_API_KEY set; otherwise try local transformers pipeline (gpt2 default),
    otherwise return a safe fallback.
    """
    # Option A: OpenAI completion/chat
    if OPENAI_API_KEY:
        try:
            openai.api_key = OPENAI_API_KEY
            # Chat completion (use gpt-3.5-turbo or gpt-4 if allowed)
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception as e:
            print("OpenAI API call failed:", e)
            # fall through to local fallback

    # Option B: local transformers
    gen = await asyncio.to_thread(get_local_generator, "gpt2", "auto")
    if gen:
        try:
            outputs = await asyncio.to_thread(gen, prompt, max_length=512, do_sample=True, num_return_sequences=1)
            # outputs is list of dicts with 'generated_text'
            text = outputs[0]["generated_text"]
            # post-process: remove the prompt prefix if present
            if text.startswith(prompt):
                text = text[len(prompt) :]
            return text.strip()
        except Exception as e:
            print("Local transformers generation failed:", e)

    # Fallback: simple echo/responder
    return f"I heard: {prompt[:100]}. (Fallback response — no LLM available.)"


# --- Step 3: TTS with edge-tts ---
async def tts_with_edge(text: str, out_wav: Path):
    """
    Use edge-tts to synthesize `text` -> save to out_wav (wav).
    Uses Microsoft voices by default; change voice as needed.
    """
    voice = "en-US-AriaNeural"  # change to preferred voice
    communicate = edge_tts.Communicate(text, voice)
    # edge-tts supports saving to file via streaming; use save() helper
    await communicate.save(str(out_wav))


# --- Endpoint: /chat/ ---
@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...), session_id: Optional[str] = Header(None)):
    # session_id header (client should provide a stable id). If not provided, create one and return to client.
    if not session_id:
        session_id = str(uuid.uuid4())

    # Save uploaded audio to tmp file
    try:
        audio_path = save_upload_file_tmp(file, TMP_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # 1) ASR
    try:
        # choose model name; change to "medium" or "large-v2" as needed
        asr_text = await run_asr(audio_path, model_name="small", device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        # cleanup
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

    # Add user message to memory
    add_to_memory(session_id, "user", asr_text)

    # 2) Generate response using LLM
    # build prompt using memory
    prompt = build_prompt_from_memory(session_id, asr_text)
    try:
        llm_response = await generate_response(prompt)
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    # Add assistant response to memory
    add_to_memory(session_id, "assistant", llm_response)

    # 3) TTS -> produce response.wav
    try:
        # remove previous response if exists
        if RESPONSE_AUDIO.exists():
            RESPONSE_AUDIO.unlink()
        await tts_with_edge(llm_response, RESPONSE_AUDIO)
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    # cleanup uploaded audio
    audio_path.unlink(missing_ok=True)

    # Return the response audio and session_id in header/body so client can maintain memory
    headers = {"X-Session-ID": session_id}
    # FileResponse will stream the audio file
    return FileResponse(str(RESPONSE_AUDIO), media_type="audio/wav", headers=headers)


@app.get("/memory/")
async def get_memory(session_id: str):
    if session_id not in CONV_MEMORY:
        return JSONResponse({"session_id": session_id, "memory": []})
    return JSONResponse({"session_id": session_id, "memory": CONV_MEMORY[session_id]})


if __name__ == "__Class3HomewokSamuel__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

### End of Step 1: FastAPI app 

### Step 2: Verify Faster-Whisper uses GPU
#python - <<'EOF'
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Set the token for Hugging Face
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# Read token from environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    print("Error: HF_TOKEN environment variable not set.")
    # Handle the error, e.g., exit or prompt for input
else:
    os.environ["HF_TOKEN"] = hf_token

from faster_whisper import WhisperModel
model = WhisperModel("small", device="cuda")
segments, info = model.transcribe("C:/Users/ch939/Downloads/LLMBootCampCodes/TestData/Audio/Oser.mp3")
print("Detected language:", info.language)
for seg in segments:
    print(f"[{seg.start:.2f} - {seg.end:.2f}]: {seg.text}")
#EOF
# End of Step 2: Verify Faster-Whisper uses GPU


### Step 3: Verify OpenAI API works
# test_setup.py
import torch
import transformers
from transformers import pipeline

print("✅ PyTorch Version:", torch.__version__)
print("✅ Transformers Version:", transformers.__version__)

print("✅ CUDA Available:", torch.cuda.is_available())
print("✅ CUDA Version (PyTorch):", torch.version.cuda)
print("✅ cuDNN Enabled:", torch.backends.cudnn.enabled)

if torch.cuda.is_available():
    print("🎮 GPU Device:", torch.cuda.get_device_name(0))
    print("  CUDA Capability:", torch.cuda.get_device_capability(0))
else:
    print("❌ GPU Not Detected!")

# Optional: Test loading a small pipeline
try:
    print("\n🔄 Testing small pipeline (distilgpt2)...")
    pipe = pipeline("text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1)
    output = pipe("Hello, I am", max_new_tokens=10)
    print("✅ Pipeline test passed:", output)
except Exception as e:
    print("❌ Pipeline test failed:", str(e))

from transformers import pipeline
hf_token = os.getenv("HUGGINGFACE_TOKEN")

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=0,  # Use GPU
    torch_dtype=torch.float16,  # Save VRAM
    token = hf_token # = os.getenv("HUGGINGFACE_TOKEN")
)    # meta-llama/Llama-3-8B-Instruct is not a public model — it won’t show up in general search

### END OF STEP 3: Verify OpenAI API works