# 🎙️ Local Voice Assistant – LLM Boot Camp Week 3

A real-time voice chatbot built using FastAPI, Hugging Face Transformers, and GPU-accelerated audio processing. This project demonstrates end-to-end speech interaction: **Speech-to-Text → LLM Response → Text-to-Speech**, with conversational memory.

🚀 *Run entirely locally — no external APIs required.*

---

## 🎯 Project Goal

Build a **local voice assistant** that can:
1. Accept audio input via HTTP (e.g., `.mp3`, `.wav`)
2. Transcribe speech to text (ASR) using **faster-whisper**
3. Generate intelligent responses using a **local LLM** (e.g., Zephyr-7B)
4. Convert responses back to speech (TTS) using **Coqui TTS**
5. Maintain **5-turn conversational memory** for context-aware replies

All processing runs on your machine using GPU acceleration (CUDA).

---

## 🧩 Features

- ✅ **Real-time audio processing** via FastAPI
- ✅ **Automatic Speech Recognition (ASR)** with `faster-whisper`
- ✅ **Local LLM inference** with `transformers` + `accelerate`
- ✅ **Text-to-Speech (TTS)** using Coqui TTS
- ✅ **Conversational memory** (last 5 exchanges)
- ✅ **GPU acceleration** (CUDA + FP16 support)
- ✅ **No cloud dependencies** — fully offline-capable

---

## 🖥️ Tech Stack

| Component        | Technology Used |
|------------------|-----------------|
| Backend          | FastAPI         |
| ASR              | [faster-whisper](https://github.com/guillaumekln/faster-whisper) |
| LLM              | Hugging Face Transformers (e.g., `zephyr-7b-beta`) |
| TTS              | [Coqui TTS](https://github.com/coqui-ai/TTS) |
| GPU Acceleration | PyTorch + CUDA 12.1 |
| Package Manager  | Conda + pip     |
| Deployment       | Uvicorn (local server) |

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/LLMBootCampCodes.git
cd LLMBootCampCodes/Week3