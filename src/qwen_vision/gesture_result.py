from config import QWEN_GESTURE_LABELS


_GESTURE_ALIASES = {
    "thumb": {
        "thumb",
        "thumbs",
        "thumbsup",
        "thumbs up",
        "thumb up",
        "thumbs-up",
        "thumb-up",
        "竖大拇指",
        "大拇指",
        "点赞",
    },
    "palm": {
        "palm",
        "open palm",
        "open hand",
        "hand palm",
        "stop",
        "手掌",
        "张开手掌",
        "摊开手掌",
        "停止",
    },
    "fist": {
        "fist",
        "closed fist",
        "拳头",
        "握拳",
    },
    "victory": {
        "victory",
        "v",
        "v sign",
        "peace",
        "剪刀手",
        "胜利",
    },
}


def build_gesture_prompt(base_prompt, labels=QWEN_GESTURE_LABELS):
    label_text = _format_label_list(labels)
    return (
        f"{base_prompt} "
        f"The gesture only can be one of: {label_text}. "
        "Reply with exactly one lowercase word from that list. "
        "Do not include punctuation, explanation, or any other text."
    )


def parse_gesture_index(text, labels=QWEN_GESTURE_LABELS):
    label = normalize_gesture_label(text, labels)
    return labels.index(label)


def normalize_gesture_label(text, labels=QWEN_GESTURE_LABELS):
    normalized = _normalize_text(text)
    label_set = set(labels)

    if normalized in label_set:
        return normalized

    for label, aliases in _GESTURE_ALIASES.items():
        if label in label_set and normalized in aliases:
            return label

    for label, aliases in _GESTURE_ALIASES.items():
        if label not in label_set:
            continue
        for alias in aliases:
            if alias and alias in normalized:
                return label

    return labels[-1]


def gesture_label_from_index(index, labels=QWEN_GESTURE_LABELS):
    if index < 0 or index >= len(labels):
        return labels[-1]
    return labels[index]


def _normalize_text(text):
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return normalized.strip(" \t\r\n.,;:!?\"'`，。；：！？（）()[]{}")


def _format_label_list(labels):
    if len(labels) == 1:
        return f"'{labels[0]}'"
    return ", ".join(f"'{label}'" for label in labels[:-1]) + f" or '{labels[-1]}'"
