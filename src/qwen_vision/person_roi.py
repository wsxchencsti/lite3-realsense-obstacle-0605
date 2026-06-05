def crop_person_roi(frame, person, pad_ratio=0.25):
    """Crop a padded person ROI from a BGR frame."""

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = person.bbox
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(frame_w, x2 + pad_x)
    crop_y2 = min(frame_h, y2 + pad_y)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if roi.size == 0:
        return None
    return roi


def person_identity(person):
    return person.track_id if person.track_id is not None else person.display_id
