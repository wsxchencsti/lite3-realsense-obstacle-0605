from dataclasses import dataclass
from datetime import datetime
import threading
import time

from config import (
    CAMERA_RESULT_DIR,
    QWEN_INTENT_INTERVAL_SECONDS,
    QWEN_PERSON_TOP_K,
    QWEN_ROI_PAD_RATIO,
    SAVE_PIC,
)
from .call_qwen_api import analyze_gesture
from .gesture_result import gesture_label_from_index
from .image_utils import encode_frame
from .person_roi import crop_person_roi, person_identity


@dataclass(frozen=True)
class PersonGestureCandidate:
    track_id: int
    display_id: int
    bbox: tuple
    confidence: float = 0.0


@dataclass(frozen=True)
class PersonGestureResult:
    person_id: int
    gesture_index: int
    gesture_label: str
    confidence: float = 0.0
    request_tag: str = ""


class PersonGestureProcessor:
    """Periodically classify person ROIs without blocking the robot loop."""

    def __init__(
        self,
        interval_seconds=QWEN_INTENT_INTERVAL_SECONDS,
        roi_pad_ratio=QWEN_ROI_PAD_RATIO,
        top_k=QWEN_PERSON_TOP_K,
    ):
        self.interval_seconds = interval_seconds
        self.roi_pad_ratio = roi_pad_ratio
        self.top_k = top_k
        self.last_result = None
        self.last_error = None
        self.last_saved_path = None
        self.last_candidate_count = 0
        self.last_request_tag = ""
        self._lock = threading.Lock()
        self._worker = None
        self._is_processing = False
        self._last_attempt_at = None
        self._current_submitted_at = None
        self._last_completed_submitted_at = None
        self._last_latency_seconds = None
        self._skipped_while_busy = 0
        self._result_version = 0
        self._consumed_result_version = 0

    def process_if_due(
        self,
        frame,
        people,
        priority_labels=None,
        now=None,
        top_k=None,
        request_tag="",
    ):
        """Submit current person ROIs when due and no request is in flight."""

        now = time.monotonic() if now is None else now
        people = _select_people(people, self.top_k if top_k is None else top_k)
        priority_labels = tuple(priority_labels or ())

        with self._lock:
            if (
                self._last_attempt_at is not None
                and now - self._last_attempt_at < self.interval_seconds
            ):
                return False

            self._last_attempt_at = now

            if self._is_processing:
                self._skipped_while_busy += 1
                busy_for = None
                if self._current_submitted_at is not None:
                    busy_for = now - self._current_submitted_at
                _log_api_status(
                    "WAIT busy tag={} busy={} skipped={}".format(
                        self.last_request_tag or "--",
                        "--" if busy_for is None else _format_seconds(busy_for),
                        self._skipped_while_busy,
                    )
                )
                return False

            self._is_processing = True
            self._current_submitted_at = now
            self._skipped_while_busy = 0
            self.last_candidate_count = len(people)
            self.last_request_tag = request_tag

        _log_api_status(
            "WAIT submit tag={} candidates={} ids={} priority={}".format(
                request_tag or "--",
                len(people),
                _format_people(people),
                ",".join(priority_labels) if priority_labels else "--",
            )
        )

        worker = threading.Thread(
            target=self._process_worker,
            args=(frame.copy(), people, priority_labels, now, request_tag),
            daemon=True,
        )
        with self._lock:
            self._worker = worker
        worker.start()
        return True

    def get_new_gesture_result(self):
        """Return each newly completed attributed result once."""

        with self._lock:
            if self._consumed_result_version == self._result_version:
                return None

            self._consumed_result_version = self._result_version
            return self.last_result

    def stop(self, timeout=2):
        with self._lock:
            worker = self._worker

        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def _process_worker(self, frame, people, priority_labels, submitted_at, request_tag):
        result = None
        error = None
        saved_path = None

        try:
            results = []
            for person in people:
                roi = crop_person_roi(frame, person, self.roi_pad_ratio)
                if roi is None:
                    _log_api_status(
                        "SKIP tag={} id={} reason=empty_roi".format(
                            request_tag or "--",
                            person_identity(person),
                        )
                    )
                    continue

                image_bytes = encode_frame(roi)
                if SAVE_PIC:
                    saved_path = save_image(image_bytes, person_identity(person))

                _log_api_status(
                    "SEND tag={} id={} conf={:.2f}".format(
                        request_tag or "--",
                        person_identity(person),
                        float(getattr(person, "confidence", 0.0)),
                    )
                )
                gesture_index = analyze_gesture(image_bytes)
                gesture_label = gesture_label_from_index(gesture_index)
                candidate = PersonGestureResult(
                    person_id=person_identity(person),
                    gesture_index=gesture_index,
                    gesture_label=gesture_label,
                    confidence=float(getattr(person, "confidence", 0.0)),
                    request_tag=request_tag,
                )
                results.append(candidate)

                if gesture_label in priority_labels:
                    break

            result = _select_result(results, priority_labels)
        except Exception as exc:
            error = str(exc)
        finally:
            finished_at = time.monotonic()
            latency_seconds = finished_at - submitted_at
            with self._lock:
                if saved_path is not None:
                    self.last_saved_path = saved_path
                if error is None:
                    self.last_result = result
                    self.last_error = None
                    if result is not None:
                        self._result_version += 1
                else:
                    self.last_error = error

                self._last_completed_submitted_at = submitted_at
                self._last_latency_seconds = latency_seconds
                self._current_submitted_at = None
                self._is_processing = False

            if error is not None:
                _log_api_status(
                    "ERROR tag={} latency={} error={}".format(
                        request_tag or "--",
                        _format_seconds(latency_seconds),
                        error,
                    )
                )
            elif result is None:
                _log_api_status(
                    "RETURN tag={} result=none latency={}".format(
                        request_tag or "--",
                        _format_seconds(latency_seconds),
                    )
                )
            else:
                _log_api_status(
                    "RETURN tag={} id={} gesture={} index={} conf={:.2f} latency={}".format(
                        request_tag or "--",
                        result.person_id,
                        result.gesture_label,
                        result.gesture_index,
                        result.confidence,
                        _format_seconds(latency_seconds),
                    )
                )

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
            candidate_count = self.last_candidate_count
            request_tag = self.last_request_tag

        return [
            _format_gesture(last_result, last_error),
            f"intent candidates: {candidate_count} tag {request_tag or '--'}",
            _format_age(now, last_completed_submitted_at),
            _format_latency(last_latency_seconds),
            _format_busy(now, is_processing, current_submitted_at, skipped_while_busy),
        ]


def _select_people(people, top_k):
    selected = list(people or [])
    selected.sort(key=lambda person: float(getattr(person, "confidence", 0.0)), reverse=True)
    if top_k is not None and top_k > 0:
        selected = selected[:top_k]
    return selected


def _select_result(results, priority_labels):
    if not results:
        return None

    for label in priority_labels:
        for result in results:
            if result.gesture_label == label:
                return result

    for result in results:
        if result.gesture_label != "other":
            return result

    return results[0]


def save_image(image_bytes, person_id):
    CAMERA_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = CAMERA_RESULT_DIR / f"person_{person_id}_{timestamp}.jpg"
    output_path.write_bytes(image_bytes)
    print(f"saved: {output_path}")
    return output_path


def _format_gesture(last_result, last_error):
    if last_error is not None:
        return f"intent: API error: {last_error}"
    if last_result is not None:
        return (
            "intent: "
            f"{last_result.gesture_label} ({last_result.gesture_index}) "
            f"id {last_result.person_id} conf {last_result.confidence:.2f}"
        )
    return "intent: waiting"


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


def _format_seconds(seconds):
    return f"{max(0.0, seconds):.1f}s"


def _format_people(people):
    parts = []
    for person in people:
        parts.append(
            "{}:{:.2f}".format(
                person_identity(person),
                float(getattr(person, "confidence", 0.0)),
            )
        )
    return "[" + ", ".join(parts) + "]"


def _log_api_status(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[QWEN_API][{timestamp}] {message}", flush=True)
