import base64

import cv2


def encode_frame(frame, image_format=".jpg"):
    """Encode one BGR frame to bytes for saving or API requests."""
    ok, encoded_image = cv2.imencode(image_format, frame)
    if not ok:
        raise RuntimeError(f"failed to encode frame as {image_format}")
    return encoded_image.tobytes()


def build_base64_image_url(image_bytes, image_format="jpeg"):
    """ 图片转base64 URL，直接送入API. """
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = f"image/{image_format.lower().lstrip('.')}"
    return f"data:{mime_type};base64,{image_base64}"
