import base64
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


OLLAMA_URL = os.getenv("QADAMCARE_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_TEXT_MODEL = os.getenv("QADAMCARE_OLLAMA_TEXT_MODEL", "qwen2.5:3b")
OLLAMA_VISION_MODEL = os.getenv("QADAMCARE_OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
TEXT_CONTEXT = int(os.getenv("QADAMCARE_TEXT_CONTEXT", "4096"))
VISION_CONTEXT = int(os.getenv("QADAMCARE_VISION_CONTEXT", "4096"))

_CPU_FIX = (
    "Ollama's model runner crashed while using CUDA. This is outside Streamlit. "
    "Quit the Ollama tray application, open PowerShell, and start Ollama in CPU mode with: "
    "$env:OLLAMA_LLM_LIBRARY='cpu_avx2'; $env:CUDA_VISIBLE_DEVICES='-1'; ollama serve. "
    "Keep that terminal open, then test the model in a second terminal."
)


def _image_to_base64(image_path, max_size=448):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _friendly_ollama_error(kind, response):
    body = response.text.strip()
    lower = body.lower()

    if "device kernel image is invalid" in lower or "cuda error" in lower:
        return RuntimeError(f"Ollama {kind} request failed because the CUDA runner crashed. {_CPU_FIX}")

    if "exceed_context_size_error" in lower or "exceeds the available context size" in lower:
        context = TEXT_CONTEXT if kind == "text" else VISION_CONTEXT
        return RuntimeError(
            f"Ollama {kind} request exceeded the active context window. "
            f"QadamCare requested {context} tokens, but the running Ollama model/server allocated less. "
            "Restart Ollama after pulling the latest project update. You can also start it with "
            "$env:OLLAMA_CONTEXT_LENGTH='4096'; ollama serve."
        )

    if "not found" in lower and "model" in lower:
        model = OLLAMA_TEXT_MODEL if kind == "text" else OLLAMA_VISION_MODEL
        return RuntimeError(
            f"Ollama {kind} model '{model}' is not installed. Run: ollama pull {model}"
        )

    return RuntimeError(
        f"Ollama {kind} request failed: HTTP {response.status_code}. {body[:1000]}"
    )


def _post_chat(payload, kind):
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=1200)
    except requests.ConnectionError as error:
        raise RuntimeError(
            "Ollama is not reachable at 127.0.0.1:11434. Start it with 'ollama serve' "
            "and keep that terminal open."
        ) from error
    except requests.Timeout as error:
        raise RuntimeError(
            "Ollama did not finish within 20 minutes. The selected model may be too large "
            "for the available RAM/VRAM."
        ) from error

    if not response.ok:
        raise _friendly_ollama_error(kind, response)

    try:
        data = response.json()
        return data["message"]["content"].strip()
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError("Ollama returned an unexpected response format.") from error


def check_ollama_status():
    base_url = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        installed = [item.get("name") for item in response.json().get("models", [])]
        return {
            "available": True,
            "installed_models": installed,
            "text_model": OLLAMA_TEXT_MODEL,
            "vision_model": OLLAMA_VISION_MODEL,
            "text_context": TEXT_CONTEXT,
            "vision_context": VISION_CONTEXT,
        }
    except Exception as error:
        return {
            "available": False,
            "installed_models": [],
            "text_model": OLLAMA_TEXT_MODEL,
            "vision_model": OLLAMA_VISION_MODEL,
            "text_context": TEXT_CONTEXT,
            "vision_context": VISION_CONTEXT,
            "message": str(error),
        }


def ask_ollama_text(prompt):
    payload = {
        "model": OLLAMA_TEXT_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": str(prompt),
            }
        ],
        "options": {
            "temperature": 0.15,
            "num_ctx": TEXT_CONTEXT,
            "num_predict": 1600,
        },
    }
    return _post_chat(payload, "text")


def ask_ollama_vision(prompt, image_paths):
    clean_paths = []
    for path in image_paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists():
            clean_paths.append(path)

    if not clean_paths:
        raise ValueError("No valid image paths were provided to Ollama vision.")

    images = [_image_to_base64(path) for path in clean_paths[:1]]
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": str(prompt),
                "images": images,
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_ctx": VISION_CONTEXT,
            "num_predict": 600,
        },
    }
    return _post_chat(payload, "vision")
