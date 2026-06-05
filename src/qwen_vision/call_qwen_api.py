import os

from config import (
    QWEN_BASE_URL,
    QWEN_GESTURE_LABELS,
    QWEN_GESTURE_PROMPT,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
)
from .gesture_result import build_gesture_prompt, parse_gesture_index
from .image_utils import build_base64_image_url


_client = None


def get_client():
    """Create and cache the DashScope OpenAI-compatible client."""
    global _client

    if _client is not None:
        return _client

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for qwen_vision") from exc

    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    _client = OpenAI(
        api_key=api_key,
        base_url=QWEN_BASE_URL,
    )
    return _client


def analyze_gesture(image_bytes, model=QWEN_MODEL):
    """Send one cropped BGR/JPEG image to Qwen-VL and return the gesture index."""
    client = get_client()
    image_url = build_base64_image_url(image_bytes)
    prompt = build_gesture_prompt(QWEN_GESTURE_PROMPT, QWEN_GESTURE_LABELS)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        extra_body={
            "enable_thinking": False,
        },
        max_tokens=QWEN_MAX_TOKENS,
        temperature=QWEN_TEMPERATURE,
    )

    raw_text = completion.choices[0].message.content.strip()
    return parse_gesture_index(raw_text, QWEN_GESTURE_LABELS)
