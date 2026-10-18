"""The ONLY file that talks to a vision AI. To swap models, change VISION_PROVIDER / VISION_MODEL in .env."""
import base64, json, os, re
from pathlib import Path

import cv2
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

PROVIDER = os.getenv("VISION_PROVIDER", "anthropic")
MODEL = os.getenv("VISION_MODEL", "claude-sonnet-5-5")
MOCK = os.getenv("MOCK_VISION", "0") == "1"


def _jpeg_bytes(image_path, max_side=1600):
    """Shrink big scans so calls are fast and cheap."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _mock(image_path, prompt):
    """Fake AI answers so you can test the whole pipeline with no API key."""
    if prompt.startswith("[COVER]"):
        rows = [{"q": q, "marks": m, "bbox": [0.1, 0.30 + i * 0.08, 0.3, 0.06]}
                for i, (q, m) in enumerate([("1a", 4), ("1b", 5), ("14b", 3)])]
        recorded = 10 if "S001" in str(image_path) else 12   # S001 has a deliberate wrong total
        return json.dumps({"rows": rows, "recorded_total": recorded,
                           "total_bbox": [0.7, 0.9, 0.15, 0.05]})
    m = re.search(r"page_(\d+)", str(image_path))
    n = int(m.group(1)) if m else 1
    q = {1: "1a", 2: "14b"}.get(n, f"{n}")
    from redink import load_page, red_pixels   # fake "AI" that sees ticks via the red-ink counter
    bbox = [0.08, 0.25, 0.84, 0.30]
    marked = red_pixels(load_page(image_path), bbox) >= 30
    return json.dumps({"regions": [{"q": q, "bbox": bbox, "has_student_writing": True, "has_evaluator_mark": marked}]})


def call_vision(image_path, prompt):
    """Send ONE image + prompt to the vision AI, get back plain text."""
    if MOCK:
        return _mock(image_path, prompt)
    data = _jpeg_bytes(image_path)
    if PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        r = client.messages.create(
            model=MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(data).decode()}},
                {"type": "text", "text": prompt}]}])
        return r.content[0].text
    if PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client()  # reads GEMINI_API_KEY
        r = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=data, mime_type="image/jpeg"), prompt])
        return r.text
    raise ValueError(f"Unknown VISION_PROVIDER: {PROVIDER}")


def call_vision_json(image_path, prompt, retries=2):
    """Same as call_vision but returns a parsed dict. Retries when the AI sends bad JSON."""
    last = ""
    for attempt in range(retries + 1):
        p = prompt if attempt == 0 else prompt + "\n\nYour last reply was not valid JSON. Reply with ONLY the JSON object, nothing else."
        last = call_vision(image_path, p)
        text = re.sub(r"```(?:json)?", "", last).strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Vision AI returned bad JSON after {retries + 1} tries: {last[:200]}")
