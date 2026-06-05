from pathlib import Path


# =================== qwen_vision/API 相关 =================== #
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.6-flash"
QWEN_GESTURE_LABELS = ("fist", "palm", "thumb", "victory", "other")
QWEN_GESTURE_PROMPT = "Identify the person's hand gesture in the image."
QWEN_MAX_TOKENS = 50
QWEN_TEMPERATURE = 0.1

# API 调度：待机时每 1s 发送 YOLO 置信度最高的 top5 人体 ROI。
CAPTURE_INTERVAL_SECONDS = 1.0
QWEN_INTENT_INTERVAL_SECONDS = CAPTURE_INTERVAL_SECONDS
QWEN_PERSON_TOP_K = 5
QWEN_ROI_PAD_RATIO = 0.25

# API 图片调试保存。
CAMERA_RESULT_DIR = Path(".temp/camera_results")
SAVE_PIC = False

# API 意图标签。
AUTONOMOUS_INTENT_ENABLED = True
TRACKING_ENTER_GESTURE = "thumb"
TRACKING_EXIT_GESTURE = "victory"
TRACKING_ENTER_CONFIRMATIONS = 2
