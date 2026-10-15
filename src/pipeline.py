import numpy as np

from src.detector import detect, to_pil
from src.postprocess import detections_to_text, draw_detections


def run(image_input, mode="auto"):
    img = to_pil(image_input)
    original = np.asarray(img)
    result = detect(img, mode=mode)
    dets = result.get("detections", [])
    text = result.get("text")
    if text is None:
        text = detections_to_text(dets)
    return {
        "detections": dets,
        "text": text,
        "annotated_image": draw_detections(original, dets),
        "original_image": original,
        "method": result.get("method", mode),
        "quality": result.get("quality", 0.0),
        "params": result.get("params", {}),
        "notes": result.get("notes", []),
    }
