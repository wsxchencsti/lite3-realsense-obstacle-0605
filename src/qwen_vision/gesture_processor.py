from datetime import datetime
import threading
import time

from config import (
    CAMERA_RESULT_DIR,
    CAPTURE_INTERVAL_SECONDS,
    SAVE_PIC,
)
from .image_utils import encode_frame
from .call_qwen_api import analyze_gesture
from .gesture_result import gesture_label_from_index


class GestureProcessor:
    """Periodically submit the current BGR frame to the API without blocking display."""

    def __init__(self, interval_seconds=CAPTURE_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds
        self.last_result = None
        self.last_error = None
        self.last_saved_path = None
        self._lock = threading.Lock()
        self._worker = None
        self._is_processing = False
        self._last_attempt_at = None
        self._current_submitted_at = None
        self._last_completed_submitted_at = None
        self._last_finished_at = None
        self._last_latency_seconds = None
        self._skipped_while_busy = 0
        self._result_version = 0
        self._consumed_result_version = 0

    def process_if_due(self, frame, now=None):
        """Submit this frame only when due and no API request is already in flight."""
        now = time.monotonic() if now is None else now

        with self._lock:
            if (
                self._last_attempt_at is not None
                and now - self._last_attempt_at < self.interval_seconds
            ):
                return False

            self._last_attempt_at = now

            if self._is_processing:
                self._skipped_while_busy += 1
                return False

            self._is_processing = True
            self._current_submitted_at = now
            self._skipped_while_busy = 0

        worker_frame = frame.copy()
        worker = threading.Thread(
            target=self._process_worker,
            args=(worker_frame, now),
            daemon=True,
        )
        with self._lock:
            self._worker = worker
        worker.start()
        return True

    def process(self, frame):
        """Synchronously save and analyze one selected frame."""
        submitted_at = time.monotonic()
        with self._lock:
            self._is_processing = True
            self._current_submitted_at = submitted_at
            self._last_attempt_at = submitted_at

        self._process_frame(frame, submitted_at)

    def get_new_gesture_result(self):
        """Return each newly completed API gesture result once."""
        with self._lock:
            if self._consumed_result_version == self._result_version:
                return None

            self._consumed_result_version = self._result_version
            return self.last_result

    def stop(self, timeout=2):
        """Wait briefly for an in-flight API request during application shutdown."""
        with self._lock:
            worker = self._worker

        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def _process_worker(self, frame, submitted_at):
        self._process_frame(frame, submitted_at)

    def _process_frame(self, frame, submitted_at):
        result = None
        error = None
        saved_path = None

        try:
            image_bytes = encode_frame(frame)
            if SAVE_PIC:
                saved_path = save_image(image_bytes)
            result = analyze_gesture(image_bytes)
            # print(f"gesture: {result}")
        except Exception as exc:
            error = str(exc)
            print(f"api frame processing failed: {exc}")
        finally:
            finished_at = time.monotonic()
            with self._lock:
                if saved_path is not None:
                    self.last_saved_path = saved_path
                if error is None:
                    self.last_result = result
                    self.last_error = None
                    self._result_version += 1
                else:
                    self.last_error = error

                self._last_completed_submitted_at = submitted_at
                self._last_finished_at = finished_at
                self._last_latency_seconds = finished_at - submitted_at
                self._current_submitted_at = None
                self._is_processing = False

    @property
    def status_text(self):
        return _join_status_parts(*self.status_lines)

    @property
    def status_lines(self):
        now = time.monotonic()
        with self._lock:
            is_processing = self._is_processing
            current_submitted_at = self._current_submitted_at
            last_result = self.last_result
            last_error = self.last_error
            last_completed_submitted_at = self._last_completed_submitted_at
            last_latency_seconds = self._last_latency_seconds
            skipped_while_busy = self._skipped_while_busy

        gesture_text = _format_gesture(last_result, last_error)
        age_text = _format_age(now, last_completed_submitted_at)
        latency_text = _format_latency(last_latency_seconds)
        busy_text = _format_busy(
            now,
            is_processing,
            current_submitted_at,
            skipped_while_busy,
        )

        return [
            gesture_text,
            age_text,
            latency_text,
            busy_text,
        ]


def _format_gesture(last_result, last_error):
    if last_error is not None:
        return f"gesture: API error: {last_error}"
    if last_result is not None:
        return f"gesture: {gesture_label_from_index(last_result)} ({last_result})"
    return "gesture: waiting"


def _format_age(now, submitted_at):
    if submitted_at is None:
        return "Age: --"
    return f"Age: {_format_seconds(now - submitted_at)}"


def _format_latency(latency_seconds):
    if latency_seconds is None:
        return "API latency: --"
    return f"API latency: {_format_seconds(latency_seconds)}"


def _format_busy(now, is_processing, current_submitted_at, skipped_while_busy):
    if not is_processing or current_submitted_at is None:
        return "API busy: idle"

    busy_text = f"API busy: {_format_seconds(now - current_submitted_at)}"
    if skipped_while_busy:
        busy_text += f", skipped {skipped_while_busy}"
    return busy_text


def save_image(image_bytes):
    CAMERA_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = CAMERA_RESULT_DIR / f"camera_{timestamp}.jpg"
    output_path.write_bytes(image_bytes)
    print(f"saved: {output_path}")
    return output_path


def _format_seconds(seconds):
    return f"{max(0.0, seconds):.1f}s"


def _join_status_parts(*parts):
    return " | ".join(part for part in parts if part)
