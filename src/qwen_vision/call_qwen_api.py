import os

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    QWEN_BASE_URL,
    QWEN_GESTURE_LABELS,
    QWEN_GESTURE_PROMPT,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
)
from .image_utils import build_base64_image_url
from .gesture_result import build_gesture_prompt, parse_gesture_index


_client = None


def get_client():
    """创建并缓存 openai client."""
    global _client

    if _client is None:
        load_dotenv()
        _client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=QWEN_BASE_URL,
        )

    return _client


def analyze_gesture(image_bytes, model=QWEN_MODEL):
    """将图片送入 api 分析手势，返回文本结果."""
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
