import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:3b"


def _image_to_base64(image_path, max_size=512):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ask_ollama_text(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": str(prompt),
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_ctx": 1024,
            "num_predict": 500,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600,
    )

    if not response.ok:
        raise RuntimeError(
            f"Ollama text request failed: {response.status_code} | {response.text}"
        )

    return response.json()["message"]["content"].strip()


def ask_ollama_vision(prompt, image_paths):
    clean_paths = []

    for path in image_paths:
        if path is None:
            continue

        path = Path(path)

        if path.exists():
            clean_paths.append(path)

    if len(clean_paths) == 0:
        raise ValueError("No valid image paths were provided to Ollama vision.")

    clean_paths = clean_paths[:1]

    images = [
        _image_to_base64(path)
        for path in clean_paths
    ]

    payload = {
        "model": OLLAMA_MODEL,
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
            "num_ctx": 1024,
            "num_predict": 500,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600,
    )

    if not response.ok:
        raise RuntimeError(
            f"Ollama vision request failed: {response.status_code} | {response.text}"
        )

    return response.json()["message"]["content"].strip()
