import argparse
import os
import sys
import threading
import time

import cv2

try:
    import tkinter as tk
except ImportError:
    tk = None

# Ensure backend models are importable when running this script directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def run_inference(frame, detector):
    """Run existing pothole inference on a single frame."""
    return detector.detect(frame)


def get_detectors():
    """Lazily import and initialize detectors.

    This avoids failing at module import time when torch / ultralytics
    are not available or incompatible with the current NumPy version.
    """
    primary_detector = None
    fallback_detector = None

    try:
        from models.pothole_detector import PotholeDetector
        primary_detector = PotholeDetector()
    except Exception as e:
        print(f"Warning: PotholeDetector unavailable: {e}")

    try:
        from models.object_detector import ObjectDetector
        fallback_detector = ObjectDetector()
    except Exception as e:
        print(f"Warning: ObjectDetector unavailable: {e}")

    return primary_detector, fallback_detector


class StreamControlWindow:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._root = None

        if tk is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        self._root = tk.Tk()
        self._root.title("Live Pothole Stream Controls")
        self._root.resizable(False, False)
        self._root.attributes("-topmost", True)

        label = tk.Label(
            self._root,
            text="Click Quit to stop the live stream",
            padx=10,
            pady=8,
        )
        label.pack()

        button = tk.Button(
            self._root,
            text="Quit",
            command=self.quit,
            width=18,
            bg="#d9534f",
            fg="white",
        )
        button.pack(padx=8, pady=(0, 10))
        self._root.protocol("WM_DELETE_WINDOW", self.quit)
        self._root.mainloop()

    def quit(self):
        self._stop_event.set()
        if self._root is not None:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass

    def should_stop(self):
        return self._stop_event.is_set()

    def close(self):
        self.quit()
        if self._thread is not None:
            self._thread.join(timeout=1)


def check_numpy_compatibility():
    try:
        import numpy as np
        major_version = int(np.__version__.split(".")[0])
        if major_version >= 2:
            print(
                "Error: NumPy 2.x is installed, but this project requires NumPy 1.x-compatible packages."
            )
            print("Please downgrade using: pip install 'numpy<2'")
            return False
    except Exception as e:
        print(f"Warning: Unable to verify NumPy compatibility: {e}")

    return True


def run_pothole_inference(frame, primary_detector, fallback_detector=None):
    """Run pothole inference using the existing model, with optional fallback."""
    potholes = []
    if primary_detector is not None:
        potholes = primary_detector.detect(frame)

    if potholes:
        return potholes

    if fallback_detector is not None:
        detections = fallback_detector.detect(frame)
        fallback_potholes = []
        for det in detections:
            class_name = det.get("class", "").lower()
            if "pothole" in class_name:
                fallback_potholes.append({
                    "bbox": det.get("bbox", []),
                    "confidence": det.get("confidence", None),
                })
        return fallback_potholes

    return []


def draw_pothole_boxes(frame, potholes, fps=None):
    """Draw red bounding boxes and labels for detected potholes."""
    for pothole in potholes or []:
        bbox = pothole.get("bbox") if isinstance(pothole, dict) else None
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        confidence = pothole.get("confidence")
        label = "Pothole"
        if confidence is not None:
            label = f"Pothole {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def create_capture(source):
    """Create OpenCV capture from webcam or video file."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video source: {source}")
    return capture


def parse_args():
    parser = argparse.ArgumentParser(description="Live pothole detection stream")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0) or path to a video file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not check_numpy_compatibility():
        return

    detector, fallback_detector = get_detectors()

    if detector is None and fallback_detector is None:
        print(
            "Error: No detector could be initialized. "
            "Ensure your environment has compatible torch/ultralytics and NumPy versions."
        )
        return

    control_window = StreamControlWindow()
    if tk is None:
        print("Info: Tkinter is not available; press 'q' in the OpenCV window to quit.")

    capture = create_capture(args.source)
    window_name = "Live Pothole Stream"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print(f"Warning: Unable to create named window: {e}")

    try:
        while True:
            start_time = time.time()

            ret, frame = capture.read()
            if not ret:
                break

            potholes = run_pothole_inference(frame, detector, fallback_detector)
            fps = 1.0 / max((time.time() - start_time), 1e-6)

            output_frame = draw_pothole_boxes(frame, potholes, fps=fps)
            cv2.imshow(window_name, output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if control_window.should_stop():
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        control_window.close()


if __name__ == "__main__":
    main()
