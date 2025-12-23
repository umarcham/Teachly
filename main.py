#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader
from subprocess import CalledProcessError, run
import tempfile
import math
from werkzeug.exceptions import BadRequest
from flask import request, jsonify, Response
import re
import json
from datetime import datetime
from pathlib import Path
from subprocess import run
from flask import request, send_file, jsonify, current_app as app


from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
import requests
from requests.auth import HTTPBasicAuth
from groq import Groq
import google.generativeai as genai


# --------------------------
# Config
# --------------------------
BASE = "https://api.d-id.com"
DID_API_KEY = os.getenv("DID_API_KEY")  # set this in your environment
DOWNLOAD_DIR = Path("clips_out")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Use env var
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not set")



# Global Sessions Store (In-Memory)
SESSIONS = {}

PIXELBIN_CLOUD = os.getenv("PIXELBIN_CLOUD")  # e.g. cool-meadow-33b2a3
PIXELBIN_KEY   = os.getenv("PIXELBIN_KEY") 
TAVUS_API_KEY = "tavus_api_key"  # set: export TAVUS_API_KEY="sk_..."
TAVUS_URL = "https://tavusapi.com/v2/conversations"   # your API Token

if not DID_API_KEY:
    raise SystemExit("Missing DID_API_KEY. Run: export DID_API_KEY='your_api_key'")

app = Flask(__name__, template_folder="templates", static_folder="assets",static_url_path="/assets")

# --------------------------
# Helpers
# --------------------------
def auth_headers():
    return {"accept": "application/json", "content-type": "application/json"}



def did_auth():
    return HTTPBasicAuth(DID_API_KEY, "")

def proxy_avatar_url(source_url):
    """
    Proxies the avatar image if it contains special characters or is from scenes-avatars.
    Downloads to static/uploads and returns a public Ngrok URL.
    """
    if not source_url or ("|" not in source_url and "scenes-avatars" not in source_url):
        return source_url

    try:
        # 1. Download image
        app.logger.info(f"Proxying avatar image: {source_url}")
        
        img_r = requests.get(source_url)
        if img_r.status_code != 200:
            # Try with auth
            img_r = requests.get(source_url, auth=did_auth())
        
        if img_r.status_code == 200:
            # 2. Save to static
            import hashlib
            ext = ".png" 
            if "image/jpeg" in img_r.headers.get("Content-Type", ""): ext = ".jpg"
            
            fname = f"proxy_{hashlib.md5(source_url.encode()).hexdigest()}{ext}"
            save_path = os.path.join("static/uploads", fname)
            os.makedirs("static/uploads", exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(img_r.content)
                
            # 3. Construct Public URL
            ngrok_url = None
            try:
                ng_r = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=1)
                if ng_r.status_code == 200:
                    tunnels = ng_r.json().get("tunnels", [])
                    for t in tunnels:
                        if t.get("proto") == "https":
                            ngrok_url = t.get("public_url")
                            break
            except Exception:
                pass
            
            if ngrok_url:
                new_url = f"{ngrok_url}/static/uploads/{fname}"
                app.logger.info(f"Proxied URL: {new_url}")
                return new_url
            else:
                # Fallback
                new_url = f"{request.host_url}static/uploads/{fname}"
                app.logger.warning(f"Could not find Ngrok URL, using: {new_url}")
                return new_url
        else:
            app.logger.error(f"Failed to download avatar for proxy: {img_r.status_code}")
            return source_url
    except Exception as e:
        app.logger.error(f"Proxy failed: {e}")
        return source_url


def create_clip(presenter_id: str = None, text: str = "", voice_id: str = None,
                result_format: str = "mp4", aspect_ratio: str = None, source_url: str = None):
    script = {"type": "text", "input": text}
    if voice_id:
        script["provider"] = {"type": "microsoft", "voice_id": voice_id}

    payload = {
        "script": script,
        "result_format": result_format,
        "config": {
            "stitch": True
        }
    }
    
    endpoint = "/clips"
    
    if presenter_id:
        payload["presenter_id"] = presenter_id
    elif source_url:
        # If source_url is provided (for custom avatars), use it
        # Proxy it first to ensure D-ID can access it
        source_url = proxy_avatar_url(source_url)
        payload["source_url"] = source_url
        endpoint = "/talks" # Use /talks for images
    else:
        raise ValueError("Either presenter_id or source_url is required")

    if aspect_ratio:
        # /talks might not support aspect_ratio in the same way, or it might.
        # D-ID /talks usually just animates the face. 
        # But let's keep it if the API supports it, or ignore it for talks if it causes error.
        # For now, we'll include it.
        payload["aspect_ratio"] = aspect_ratio

    r = requests.post(
        f"{BASE}{endpoint}",
        auth=did_auth(),
        headers=auth_headers(),
        data=json.dumps(payload),
        timeout=60
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Create clip failed [{r.status_code}]: {r.text} Payload: {json.dumps(payload)}")
    return r.json()

def get_clip(clip_id: str):
    # Try /clips first
    r = requests.get(
        f"{BASE}/clips/{clip_id}",
        auth=did_auth(),
        headers={"accept": "application/json"},
        timeout=60
    )
    if r.status_code == 404:
        # If not found, try /talks (for custom avatars)
        r = requests.get(
            f"{BASE}/talks/{clip_id}",
            auth=did_auth(),
            headers={"accept": "application/json"},
            timeout=60
        )
    
    if r.status_code >= 400:
        raise RuntimeError(f"Get clip/talk failed [{r.status_code}]: {r.text}")
    return r.json()

def get_presenters():
    r = requests.get(
        f"{BASE}/clips/presenters",
        auth=did_auth(),
        headers={"accept": "application/json"},
        timeout=60
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Get presenters failed [{r.status_code}]: {r.text}")
    data = r.json()
    # sometimes API wraps as {"presenters":[...]} — normalize
    if isinstance(data, dict) and "presenters" in data:
        return data["presenters"]
    return data

def call_gemini(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    r = requests.post(GEMINI_URL, headers=headers, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini Error: {r.text}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "No response from Gemini"


@app.post("/api/pdf-to-script")
def api_pdf_to_script():
    """
    Upload a PDF → Extract text → Simplify using Gemini → Return clean script.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No PDF uploaded"}), 400

    file = request.files["file"]
    try:
        reader = PdfReader(file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
    except Exception:
        return jsonify({"ok": False, "error": "Could not extract text from PDF"}), 500

    prompt = (
        "You are a teacher. Convert this text into a simple student-friendly spoken explanation. "
        "Keep it conversational and clear" + raw_text
    )

    try:
        simplified = call_gemini(prompt)
        return jsonify({"ok": True, "script": simplified})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/generate-whiteboard-script")
def api_generate_whiteboard_script():
    data = request.json or {}
    topic = data.get("prompt")
    if not topic:
        return jsonify({"ok": False, "error": "Prompt is required"}), 400

    system_prompt = """
    You are the Whiteboard Animation Engine for an AI Tutor video.
    Your job:
    → Generate clear, step-by-step whiteboard content that perfectly matches the teacher’s spoken explanation.
    → Do NOT write as a paragraph. Use structured diagrams, formulas, steps, arrows, and visual layout.
    → Your output will be rendered on a whiteboard next to the tutor avatar.

    Follow these rules:
    1. ALWAYS give content in simple, clean whiteboard layout.
    2. Use ASCII diagrams, arrows, boxes, and clear math formatting.
    3. Make each step appear one by one (I will animate each line).
    4. Highlight important points using **bold** or >>> arrows.
    5. Content must be optimized for teaching, not decoration.
    6. Avoid long sentences. Use teaching-style shorthand.
    7. If the topic requires diagrams, generate clear ASCII diagrams.
    8. If the topic requires equations, show them step-by-step.
    9. If the topic requires proofs, structure them like:
       - Given
       - To Prove
       - Steps
       - Final Result
    10. Keep everything within the whiteboard area (no long-width lines).

    Your output format must be:

    WHITEBOARD_TITLE: <short title>

    STEP_1:
    <first content line>

    STEP_2:
    <second content line>

    STEP_3:
    <third content line>

    (Add as many steps as needed)

    DIAGRAM (if required):
    <ASCII diagram>

    HIGHLIGHTS:
    - <key point 1>
    - <key point 2>
    """
    
    user_prompt = f"Generate whiteboard content for the following topic exactly in the above format: {topic}"
    
    try:
        # Reusing call_gemini helper but with a more specific prompt structure if needed.
        # Ideally we concatenate system + user prompt if the helper only takes one string.
        full_prompt = f"{system_prompt}\n\nYOUR TASK NOW:\n{user_prompt}"
        script = call_gemini(full_prompt)
        return jsonify({"ok": True, "script": script})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

DID_API_BASE = "https://api.d-id.com"

def fetch_did_agents():
    """Return a list of agents from D-ID. Return [] on failure."""
    api_key = os.environ.get("DID_API_KEY")
    if not api_key:
        app.logger.warning("DID_API_KEY not set")
        return []

    try:
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        # endpoint per docs: GET https://api.d-id.com/agents/me
        r = requests.get(f"{DID_API_BASE}/agents/me", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Inspect data structure — adapt to actual shape.
        # I’ll handle common shapes: {agents: [...] }  OR {data: [...]} OR direct list
        if isinstance(data, dict):
            for key in ("agents", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # if response is itself a single agent object, return it as list
            if "agent_id" in data or "id" in data:
                return [data]
        if isinstance(data, list):
            return data
    except Exception as e:
        app.logger.exception("Failed to fetch D-ID agents: %s", e)
    return []



# --- Voices (D-ID TTS) ---
def get_voices():
    """Fetch all TTS voices from D-ID and return only Microsoft voices (best for Clips)."""
    url = f"{BASE}/tts/voices"
    r = requests.get(url, auth=did_auth(), headers={"accept": "application/json"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Normalize: keep a compact record
    voices = []
    for v in data if isinstance(data, list) else []:
        if (v.get("provider") or "").lower() == "microsoft":
            voices.append({
                "voice_id": v.get("voice_id") or v.get("id"),
                "name": v.get("name") or v.get("voice_id"),
                "locale": v.get("locale"),
                "gender": v.get("gender"),
                "preview_url": v.get("preview_url") or v.get("sample_url"),
                "provider": "microsoft"
            })
    # Deduplicate by voice_id (just in case)
    seen = set(); uniq = []
    for v in voices:
        vid = v["voice_id"]
        if vid and vid not in seen:
            uniq.append(v); seen.add(vid)
    # Sort (locale then name)
    uniq.sort(key=lambda x: (x.get("locale") or "", x.get("name") or ""))

    return uniq



def make_srt_from_text(text: str, seconds_per_line: int = 2) -> str:
    """
    Very simple subtitle splitter:
    Each sentence -> one subtitle block of `seconds_per_line` seconds.
    Uses _format_ts so timestamps remain valid even past 60s.
    Returns SRT text (string).
    """
    # split on sentence end punctuation but keep sensible chunks
    chunks = [c.strip() for c in re.split(r'[.!?]+', text) if c.strip()]
    srt_lines = []
    t = 0.0
    idx = 1
    for chunk in chunks:
        start_ts = _format_ts(t)
        t += float(seconds_per_line)
        end_ts = _format_ts(t)
        # ensure a final period if missing
        line_text = chunk
        if not line_text.endswith(('.', '!', '?')):
            line_text = line_text + '.'
        srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{line_text}\n")
        idx += 1
    return "\n".join(srt_lines)


def _format_ts(t: float) -> str:
    """
    t in seconds (float) -> "HH:MM:SS,mmm"
    """
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    # handle possible rounding that makes ms = 1000
    if ms >= 1000:
        s += 1
        ms -= 1000
        if s >= 60:
            s = 0
            m += 1
            if m >= 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_from_segments(segments) -> str:
    """
    Build SRT text from Whisper-style segments.
    Each segment must have .start, .end, .text attributes (floats for times).
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_ts(float(seg.start))
        end = _format_ts(float(seg.end))
        text = (seg.text or "").strip()
        if not text:
            continue
        # normalize newlines in text
        text = text.replace("\r\n", "\n")
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def make_srt_with_whisper(audio_path, srt_path, model_size="small", language=None):
    """
    Transcribe `audio_path` with faster-whisper and write SRT to `srt_path`.
    `audio_path` and `srt_path` can be Path or str.
    """
    from faster_whisper import WhisperModel
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    # instantiate model (device='cpu' recommended for mac without CUDA)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    try:
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,              # e.g., "en" or None for auto
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=5
        )
        # materialize generator into list so we can format SRT
        segs = list(segments)
        srt_text = srt_from_segments(segs)
        # normalize newlines to LF
        srt_text = srt_text.replace("\r\n", "\n")
        srt_path.write_text(srt_text, encoding="utf-8")
        return str(srt_path)
    except Exception as e:
        # bubble up or log as needed; caller should handle fallback
        raise
    finally:
        # if the WhisperModel exposes cleanup, do it here (it typically doesn't)
        pass

@app.post("/api/tavus/conversations")
def tavus_create_conversation():
    if not TAVUS_API_KEY:
        return jsonify({"ok": False, "error": "TAVUS_API_KEY not set"}), 500

    data = request.get_json(force=True) or {}
    replica_id = (data.get("replica_id") or "").strip()
    persona_id = (data.get("persona_id") or "").strip()
    conversation_name = (data.get("conversation_name") or "").strip()

    if not replica_id:
        return jsonify({"ok": False, "error": "replica_id is required"}), 400

    payload = {"replica_id": replica_id}
    if persona_id:
        payload["persona_id"] = persona_id
    if conversation_name:
        payload["conversation_name"] = conversation_name

    try:
        r = requests.post(
            TAVUS_URL,
            headers={
                "x-api-key": TAVUS_API_KEY,
                "content-type": "application/json",
                "accept": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except requests.RequestException as e:
        app.logger.exception("Tavus upstream error")
        return jsonify({"ok": False, "error": f"Upstream request failed: {e}"}), 502

    if r.status_code >= 400:
        app.logger.error("Tavus %s: %s", r.status_code, r.text)

    return Response(r.content, status=r.status_code, content_type=r.headers.get("content-type", "application/json"))

# ----- Tavus pickers (mock data you can edit/replace) -----


@app.get("/api/tavus/replicas")
def list_replicas():
    r = requests.get(
        "https://tavusapi.com/v2/replicas",
        headers={"x-api-key": TAVUS_API_KEY},
        timeout=30
    )
    data = r.json()

    items = []
    for rep in data.get("data", []):
        items.append({
            "id": rep.get("replica_id"),
            "name": rep.get("replica_name") or rep.get("replica_id"),
            "thumbVideo": rep.get("thumbnail_video_url"),   # <— video preview
            "status": rep.get("status"),
            "progress": rep.get("training_progress"),
        })
    return {"items": items}






@app.get("/api/voices")
def api_voices():
    try:
        return jsonify({"ok": True, "voices": get_voices()})
    except Exception as e:
        # Fallback set if D-ID endpoint is blocked on your plan
        fallback = [
            {"voice_id": "en-US-JennyNeural", "name": "Jenny (US)", "locale": "en-US", "gender": "female", "preview_url": None, "provider": "microsoft"},
            {"voice_id": "en-US-GuyNeural",   "name": "Guy (US)",   "locale": "en-US", "gender": "male",   "preview_url": None, "provider": "microsoft"},
            {"voice_id": "en-IN-NeerjaNeural","name": "Neerja (IN)","locale": "en-IN", "gender": "female", "preview_url": None, "provider": "microsoft"},
            {"voice_id": "en-IN-PrabhatNeural","name": "Prabhat (IN)","locale": "en-IN","gender":"male","preview_url": None, "provider": "microsoft"},
            {"voice_id": "en-GB-LibbyNeural", "name": "Libby (UK)", "locale": "en-GB", "gender": "female", "preview_url": None, "provider": "microsoft"},
        ]
        return jsonify({"ok": True, "voices": fallback, "fallback": True})

# --------------------------
# Routes
# --------------------------
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/realtime")
def realtime_page():
    # If you want to pass variables into the template, add them as kwargs.
    return render_template("realtime.html")

@app.get("/video-call")
def video_call_page():
    return render_template("video_call.html")

@app.get("/whiteboard")
def whiteboard_page():
    return render_template("whiteboard.html")

@app.get("/askteach")
def askteach_page():
    return render_template("askteach.html")

@app.get("/code-explainer")
def code_explainer_page():
    # List available examples
    examples = []
    data_dir = os.path.join(os.getcwd(), "data", "code_examples")
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".json"):
                 examples.append(f)
    return render_template("code_explainer.html", examples=examples)

@app.get("/api/code-examples/<filename>")
def get_code_example(filename):
    # Security check: only allow alphanum and basic chars
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
        return jsonify({"error": "Invalid filename"}), 400
    
    path = os.path.join(os.getcwd(), "data", "code_examples", filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
        
    with open(path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.post("/api/code-explainer/generate")
def generate_code_visualization():
    data = request.json or {}
    code_snippet = data.get("code")
    if not code_snippet:
         return jsonify({"error": "No code provided"}), 400

    prompt = f"""
    You are a C Language Visualization Engine. 
    Convert the following C code into a JSON Steps format for a memory visualization tool.
    
    CODE:
    {code_snippet}
    
    OUTPUT FORMAT (JSON ONLY):
    {{
      "title": "Title of program",
      "language": "c",
      "code": [list of strings, one per line of code],
      "lines": [
         {{
           "line": <line_number_1_based>,
           "explanation": "Simple explanation of what happens here (beginner friendly).",
           "actions": [
              {{ "type": "createBox", "id": "varName", "label": "varName", "value": 10, "segment": "stack" }},
              {{ "type": "updateBox", "id": "varName", "value": 20 }},
              {{ "type": "updateArrayIndex", "id": "arr", "index": 0, "value": 99 }},
              {{ "type": "deleteBox", "id": "arr" }},
              {{ "type": "highlightBox", "id": "varName" }},
              {{ "type": "showModule", "label": "<stdio.h>" }}
           ]
         }}
      ]
    }}
    
    RULES:
    1. actions types allowed: createBox, updateBox, updateArrayIndex, deleteBox, highlightBox, createArray (with 'values':[]), showModule, addArrow.
    2. For arrays: createArray with initial 'values'. Use updateArrayIndex to change single elements.
    3. For malloc/calloc: Use 'createBox' or 'createArray' with "segment": "heap". 
    4. For free(ptr): Use 'deleteBox' on the ID of the *heap object* being pointed to (NOT the pointer variable itself).
    5. Function Return: Explicitly use 'deleteBox' for ALL local variables (stack cleanup) when a function returns.
    6. For POINTERS: Use 'createBox' for the pointer variable itself, THEN 'addArrow' from pointer ID to target ID.
       - Example: int *p = &n; -> createBox("p"), addArrow("p", "n").
    8. For STRUCTS:
       - Create: Use 'createBox' with "fields": {{ "key": "value", ... }}. Do NOT set "value".
       - Update Member: Use 'updateBox' with "fields": {{ "key": "newValue" }} on the struct's ID.
       - Arrays in Structs: MUST represent as a string "[v1, v2, ...]" (e.g. "[5, 15, _, _, _]"). DO NOT say "int[]".
    11. For QUEUE/STACK Removal:
       - If a value is logically removed (e.g. dequeued) but physically remains in the array, surround it with parens: "(5)".
       - Example: "[ (5), 15, 25, _, _ ]" means 5 was dequeued, 15 and 25 are active.
    12. Respond with RAW JSON. No Markdown. No ```json fences.
    13. Trace the logic carefully.
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return jsonify(json.loads(text))
    except Exception as e:
        print(f"GenAI Error: {e}")
        return jsonify({"error": "Failed to generate visualization"}), 500

@app.post("/api/askteach/chat")
def askteach_chat():
    data = request.json or {}
    user_msg = data.get("message")
    profile = data.get("profile", {})
    
    if not user_msg:
        return jsonify({"ok": False, "error": "Message required"}), 400

    # Construct System Prompt based on Profile
    # Using the advanced Teachly prompt design
    name = profile.get("name", "Student")
    current_focus = profile.get("currentFocus", "General")
    
    # Extract deeply nested preferences with defaults
    style = profile.get("stylePreferences", {})
    lang_style = style.get("languageStyle", {}).get("language", "english")
    tone = style.get("personalityTone", {}).get("mainTone", "supportive")
    pace = style.get("pace", "medium")
    
    level_info = profile.get("learningLevel", {}) 
    
    system_prompt = (
        "SYSTEM PROMPT: You are 'AskTeach', an advanced adaptive AI tutor.\n"
        "Your goal is to personalize learning based on the user's 'TeachlyUserProfile'.\n"
        "Use the JSON profile provided in every request to adapt your tone, pace, content style, and language.\n\n"
        "VISUAL DIAGRAMS (Mermaid.js):\n"
        "If a concept is complex (e.g. 'Network Topology', 'Process Flow', 'Relationship'), generate a Mermaid.js diagram.\n"
        "1. Start logic with `graph TD` or `sequenceDiagram`.\n"
        "2. **STRICT SYNTAX RULES**:\n"
        "   - **Quote ALL Labels**: checking space or special chars. Example: `A[\"label with spaces\"]` instead of `A[label with spaces]`.\n"
        "   - **Avoid Special Chars**: Do not use `()`, `[]`, or curly braces inside node text unless escaped or heavily quoted. Keep labels simple.\n"
        "   - **No Conversational Text**: The block must contain ONLY valid mermaid code. No comments like 'Here is the diagram'.\n"
        "3. Use standard `graph TD` for hierarchies and `sequenceDiagram` for flows.\n"
        "4. Wrap the code in markdown fences: ```mermaid ... ```.\n\n"
        "GENERATIVE IMAGES (HYBRID STRATEGY):\n"
        "If a visual would help (e.g. 'Show me a lion', 'What does the Eiffel Tower look like?', 'Image of flowchart'):\n"
        "- Use this exact ONE-LINE format (no newlines inside the tag):\n"
        "  <img src='https://tse2.mm.bing.net/th?q={query}&w=800&h=600' onerror=\"this.onerror=null; this.src='https://image.pollinations.ai/prompt/{query}?width=800&height=600&nologo=true';\" alt='{query}' style='border-radius:12px; width:100%; margin-top:10px; display:block;' />\n"
        "- Replace {query} with a simple search term (e.g. 'Lion', 'Eiffel Tower', 'Flowchart of tea making').\n"
        "- If user explicitly requests 'image of diagram' or 'image of flowchart' (instead of just 'flowchart'), use this image format instead of Mermaid.\n"
        "- For PURELY abstract concepts (e.g. 'Cyberpunk Soul'), use Pollinations directly.\n\n"
        "INTERACTIVE SIMULATIONS (P5.JS):\n"
        "If the user asks for a simulation (e.g. 'doppler', 'gravity', 'orbit', 'stokes law'):\n"
        "1. Generate high-quality P5.js Instance Mode code.\n"
        "2. Put the RAW CODE STRING into the 'animation_code' key of the JSON response.\n"
        "3. **VISUAL RULES (CRITICAL)**:\n"
        "   - **Draw the Environment**: Don't just draw a ball. Draw the ground, the container (beaker), the walls, or the sky.\n"
        "   - **Use Labels**: Use `p.text('Label', x, y)` to explain what things are (e.g. 'Honey', 'Drag Force').\n"
        "   - **Colors**: Use modern, vibrant colors (e.g. #4f46e5 for objects, #e0f2fe for sky, #fef3c7 for honey). NO DEFAULT GREY BACKGROUNDS.\n"
        "   - **Interaction**: Make it interactive (e.g. 'p.mouseX' controls position or variable).\n"
        "4. **Code Rules**:\n"
        "   - Use 'p.' for ALL P5 functions (p.setup, p.draw, p.fill, p.background, p.width).\n"
        "   - Do NOT wrap in markdown fences. Just the raw code string.\n"
        "   - Use `p.createCanvas(600, 400)`.\n\n"
        
        f"You are AskTeach, an adaptive AI tutor for {name}. "
        f"Your goal is to be helpful and {tone}.\n"
        f"Current Focus: {current_focus}.\n"
        f"Profile Context:\n"
        f"- Levels: {json.dumps(level_info)}\n"
        f"- Preferences: {json.dumps(style)}\n\n"

        "IMPLICIT CONTEXT & PROACTIVE VISUALS (CRITICAL):\n"
        "1. **Never ask 'Which images?' or 'What kind of diagram?'** if the user asks for 'images', 'diagrams', or 'examples' after a topic is established.\n"
        "2. **Infer the context** from the conversation history (e.g. if discussing 'TIR', show a TIR diagram).\n"
        "3. **IMMEDIATELY generate** the most appropriate visual:\n"
        "   - **Diagram (Mermaid)**: For structures, flows, or relationships.\n"
        "   - **Simulation (P5)**: For physics, motion, or interactive concepts.\n"
        "   - **Real Image (Hybrid)**: For real-world objects (e.g. 'Lion', 'DNA helix'). Use the <img src='...'> format defined above.\n"
        "4. **ALWAYS OUTPUT CODE**: If your text says 'Here is a diagram', you MUST include the ```mermaid ... ``` block. Do not forget it.\n"
        "5. **CONTEXT ADHERENCE**: If the user asks for images about 'Thermodynamics', the visual MUST be about 'Thermodynamics'.\n\n"
        
        "CRITICAL INSTRUCTION: You must respond in valid JSON format ONLY. Do not wrap in markdown.\n"
        "Your JSON object must have these keys:\n"
        "1. 'reply': Your conversational response in HTML (use <b>, <i>, <br>, <pre>). Adapt to the user's language and style.\n"
        "2. 'updates': A dictionary of profile fields to update.\n"
        "3. 'animation_code': (Optional) The raw P5.js code string if a simulation was requested. Otherwise null.\n"
        "   - DETECT LEARNING STYLE: If they ask for examples, set 'reasoningPreference': 'example-first'.\n"
        "   - DETECT PACE: If they say 'slow down', set 'pace': 'slow'.\n"
        "   - DETECT FOCUS: If topic changes (e.g. 'dopler' -> 'Physics'), update 'currentFocus'.\n"
        "   - DETECT LEVEL: If they understand well, increase confidenceScore/level in learningLevel.\n"
        "   - Example update:\n"
        "     {\n"
        "       'currentFocus': 'Physics',\n"
        "       'stylePreferences': { 'contentStyle': { 'prefersDiagrams': true } },\n"
        "       'learningLevel': { 'physics': { 'level': 'intermediate', 'confidenceScore': 0.6 } }\n"
        "     }\n"
    )
    
    history_list = data.get("history", [])
    history_str = ""
    for msg in history_list:
        role = "Student" if msg.get("sender") == "user" else "AskTeach"
        content = msg.get("text", "")
        if content:
             history_str += f"{role}: {content}\n"
    
    full_prompt = f"{system_prompt}\n\n[CONVERSATION HISTORY]\n{history_str}\n[CURRENT MESSAGE]\nStudent: {user_msg}\nAskTeach (JSON):"
    
    app.logger.info(f"--- CHAT CONTEXT DEBUG ---\nHistory:\n{history_str}\n--------------------------")
    app.logger.info(f"--- FULL PROMPT SENT TO GEMINI ---\n{full_prompt}\n------------------------------")
    
    try:
        raw_response = call_gemini(full_prompt)
        app.logger.info(f"Raw Gemini Response: {raw_response}") 

        # Robust cleanup
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"): clean_json = clean_json[7:]
        if clean_json.startswith("```"): clean_json = clean_json[3:]
        if clean_json.endswith("```"): clean_json = clean_json[:-3]
        clean_json = clean_json.strip()
        
        response_data = json.loads(clean_json)
        app.logger.info(f"Parsed Updates: {response_data.get('updates', {})}")
        
        return jsonify({
            "ok": True, 
            "reply": response_data.get("reply", ""),
            "updates": response_data.get("updates", {}),
            "animation_code": response_data.get("animation_code")
        })
    except Exception as e:
        app.logger.error(f"AskTeach Error: {e}")
        return jsonify({"ok": True, "reply": raw_response, "updates": {}})

# --- D-ID Real-time Streaming & AI Pipeline ---

@app.post("/create-stream")
def create_stream():
    """
    Initialize a D-ID stream.
    """
    
    try:
        data = request.json or {}
        user_source_url = data.get("source_url")

        if user_source_url:
            source_url = user_source_url
        else:
            # Fetch default presenters from D-ID to ensure we use a valid image
            presenters = get_presenters()
            source_url = "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg" # Fallback
            
            if presenters:
                # Try to find a presenter with a valid image_url
                for p in presenters:
                    if p.get("image_url"):
                        source_url = p["image_url"]
                        break
        
        # Fix for D-ID internal parser failing on pipes in URLs
        # Also, D-ID sometimes fails to fetch from its own scenes-avatars domain.
        # We will proxy the image: Download it -> Save to static -> Serve via Ngrok
        source_url = proxy_avatar_url(source_url)

        payload = {
            "source_url": source_url,
            "config": {
                "stitch": True
            }
        }
        headers = {
            "Authorization": f"Basic {DID_API_KEY}:", # Note the colon for Basic Auth with just API Key
            "Content-Type": "application/json"
        }
        # If using the new API format, check D-ID docs. This assumes /talks/streams
        r = requests.post(f"{BASE}/talks/streams", json=payload, headers=headers)
        if r.status_code >= 400:
            app.logger.error(f"D-ID Error: {r.status_code} - {r.text}")
            return jsonify({"error": f"D-ID Error: {r.text}"}), r.status_code
            
        return jsonify(r.json())
    except Exception as e:
        app.logger.error(f"Create stream failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/submit-network-info")
def submit_network_info():
    """
    Exchange ICE candidates/SDP with D-ID.
    """
    try:
        data = request.json
        session_id = data.get("session_id")
        stream_id = data.get("stream_id")
        
        headers = {
            "Authorization": f"Basic {DID_API_KEY}:",
            "Content-Type": "application/json"
        }
        
        # Determine if it's an answer or ICE candidate
        if "answer" in data:
            payload = {"answer": data["answer"], "session_id": session_id}
            url = f"{BASE}/talks/streams/{stream_id}/sdp"
        elif "candidate" in data:
            # Flatten the candidate object
            candidate_data = data["candidate"]
            payload = {
                "candidate": candidate_data.get("candidate"),
                "sdpMid": candidate_data.get("sdpMid"),
                "sdpMLineIndex": candidate_data.get("sdpMLineIndex"),
                "session_id": session_id
            }
            url = f"{BASE}/talks/streams/{stream_id}/ice"
        else:
            return jsonify({"error": "Invalid network info"}), 400

        r = requests.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
             app.logger.error(f"Network info failed: {r.status_code} - {r.text}")
             return jsonify({"error": f"D-ID Error: {r.text}"}), r.status_code
        r.raise_for_status()
        return jsonify({"ok": True})
    except Exception as e:
        app.logger.error(f"Network info failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/process-audio")
def process_audio():
    """
    1. Receive audio blob
    2. STT (Groq)
    3. LLM (Gemini)
    4. Send text to D-ID stream
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        audio_file = request.files["audio"]
        session_id = request.form.get("session_id")
        stream_id = request.form.get("stream_id")

        # 1. Save temp audio
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name

        # 2. STT with Groq
        try:
            with open(temp_path, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(temp_path, f.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            user_text = transcription
            app.logger.info(f"User said: {user_text}")
        
            # Filter Hallucinations
            ignore_list = ["Thank you.", "Thanks.", "Okay.", "Tchau.", "Bye.", "You.", "MBC News", "Amara.org", "Subtitle by", "Copyright"]
            cleaned_text = user_text.strip()
            
            if not cleaned_text or len(cleaned_text) < 2 or any(phrase.lower() in cleaned_text.lower() for phrase in ignore_list):
                 app.logger.info(f"Ignored hallucination/noise: {cleaned_text}")
                 return jsonify({"ok": True, "transcription": "", "reply": ""})

        except Exception as e:
            app.logger.error(f"Groq STT failed: {e}")
            return jsonify({"error": f"Groq STT failed: {e}"}), 500
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if not user_text or not user_text.strip():
             return jsonify({"ok": True, "transcription": "", "reply": ""})

        # 3. LLM with Gemini
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            chat = model.start_chat(history=[])
            response = chat.send_message(user_text)
            ai_reply = response.text.replace('*', '')
            app.logger.info(f"AI reply: {ai_reply}")
        except Exception as e:
            app.logger.error(f"Gemini LLM failed: {e}")
            return jsonify({"error": f"Gemini LLM failed: {e}"}), 500

        # 4. Send to D-ID
        if session_id and stream_id:
            try:
                headers = {
                    "Authorization": f"Basic {DID_API_KEY}:",
                    "Content-Type": "application/json"
                }
                # Simplified payload to reduce risk of errors
                payload = {
                    "script": {
                        "type": "text",
                        "input": ai_reply,
                        "provider": {
                            "type": "microsoft",
                            "voice_id": "en-US-JennyNeural"
                        }
                    },
                    "session_id": session_id
                }
                r = requests.post(f"{BASE}/talks/streams/{stream_id}", json=payload, headers=headers)
                if r.status_code >= 400:
                    app.logger.error(f"D-ID Stream Error: {r.status_code} - {r.text}")
                    return jsonify({"error": f"D-ID Stream Error: {r.text}"}), r.status_code
                r.raise_for_status()
            except Exception as e:
                app.logger.error(f"D-ID request failed: {e}")
                return jsonify({"error": f"D-ID request failed: {e}"}), 500

        return jsonify({
            "ok": True, 
            "transcription": user_text, 
            "reply": ai_reply
        })

    except Exception as e:
        app.logger.error(f"Process audio failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/test-audio")
def test_audio():
    """
    Directly send text to D-ID stream for testing.
    """
    try:
        data = request.json
        session_id = data.get("session_id")
        stream_id = data.get("stream_id")
        text = data.get("text", "Hello, this is a test.")

        if not session_id or not stream_id:
            return jsonify({"error": "Missing session_id or stream_id"}), 400

        headers = {
            "Authorization": f"Basic {DID_API_KEY}:",
            "Content-Type": "application/json"
        }
        payload = {
            "script": {
                "type": "text",
                "input": text,
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-JennyNeural"
                }
            },
            "session_id": session_id
        }
        r = requests.post(f"{BASE}/talks/streams/{stream_id}", json=payload, headers=headers)
        if r.status_code >= 400:
             app.logger.error(f"D-ID Test Error: {r.status_code} - {r.text}")
             return jsonify({"error": f"D-ID Error: {r.text}"}), r.status_code
        r.raise_for_status()

        return jsonify({"ok": True})
    except Exception as e:
        app.logger.error(f"Test audio failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/close-stream")
def close_stream():
    try:
        data = request.json
        session_id = data.get("session_id")
        stream_id = data.get("stream_id")
        
        headers = {
            "Authorization": f"Basic {DID_API_KEY}:",
            "Content-Type": "application/json"
        }
        payload = {"session_id": session_id}
        requests.delete(f"{BASE}/talks/streams/{stream_id}", json=payload, headers=headers)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/presenters")
def api_presenters():
    try:
        presenters = get_presenters()
        return jsonify({"ok": True, "presenters": presenters})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    



       

@app.post("/api/clip")
def api_create_clip():
    """
    Body: { "presenter_id": "...", "text": "...", "voice_id": "en-US-JennyNeural", "aspect_ratio": "16:9", "source_url": "..." }
    """
    data = request.get_json(force=True)
    presenter_id = data.get("presenter_id")
    source_url = data.get("source_url")
    text = data.get("text")
    voice_id = data.get("voice_id")
    aspect_ratio = data.get("aspect_ratio")
    
    if not text:
        return jsonify({"ok": False, "error": "text is required"}), 400
        
    if not presenter_id and not source_url:
        return jsonify({"ok": False, "error": "presenter_id or source_url is required"}), 400
        
    try:
        created = create_clip(
            presenter_id=presenter_id,
            source_url=source_url,
            text=text,
            voice_id=voice_id,
            aspect_ratio=aspect_ratio
        )
        return jsonify({"ok": True, "clip": created})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/clip/<clip_id>")
def api_get_clip(clip_id):
    try:
        info = get_clip(clip_id)
        return jsonify({"ok": True, "clip": info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------- Gemini translate helper ----------
def call_gemini_translate(text: str, target_lang: str):
    """
    Translate `text` into `target_lang` using your Gemini server-side key.
    Returns translated text (string). Raises on failure.
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    # Keep the prompt minimal, ask Gemini to return only the translated text
    prompt = (
        f"Translate the following text into {target_lang}.\n"
        "Return only the translated text and nothing else.\n\n"
        f"{text}"
    )
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    r = requests.post(GEMINI_URL, headers=headers, json=body, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text}")
    data = r.json()
    # adapt to Gemini response shape
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected Gemini response: {e} / {r.text}")



# ---------- helper functions (include in same module) ----------

def parse_srt_blocks(srt_text):
    """
    Parse SRT into list of dicts: [{idx, start, end, text}, ...]
    Keeps timestamps raw for re-writing.
    """
    blocks = []
    raw_blocks = [b.strip() for b in srt_text.split("\n\n") if b.strip()]
    for b in raw_blocks:
        lines = b.splitlines()
        if len(lines) < 3:
            continue
        idx = lines[0].strip()
        times = lines[1].strip()
        text = "\n".join(lines[2:]).strip()
        # keep start/end strings if needed; we just preserve times in output
        blocks.append({"idx": idx, "times": times, "text": text})
    return blocks

def build_srt_from_items(items):
    """
    items: [{idx, times, text}, ...] -> return SRT string
    """
    out = []
    for i in items:
        out.append(f"{i['idx']}\n{i['times']}\n{i['text']}\n")
    return "\n".join(out)

# at top of file (imports)
from deep_translator import GoogleTranslator

def translate_srt_items(items, target_lang):
    """
    Translate each item's text into `target_lang` using deep-translator.
    items: list of dicts: {'idx','times','text'}
    Returns new list same shape with translated text.
    """
    texts = [it["text"] for it in items]
    out = []
    # deep-translator's GoogleTranslator doesn't offer a batch API reliably,
    # so translate in a loop (small number of segments is fine).
    translator = GoogleTranslator(source="auto", target=target_lang)
    for it in items:
        try:
            translated = translator.translate(it["text"])
        except Exception as e:
            # fallback: keep original text on any error
            app.logger.error("Translate error: %s", e)
            translated = it["text"]
        new = it.copy()
        new["text"] = translated
        out.append(new)
    return out


def ffmpeg_subtitles_filter_arg(path: Path) -> str:
    """
    Build a safe subtitles filter argument for ffmpeg.
    On Windows backslashes must be doubled; on *nix it's fine.
    Returns the string used for -vf "subtitles=..."
    """
    p = str(path.resolve())
    # escape backslashes
    p = p.replace("\\", "\\\\")
    # wrap in single quotes to allow spaces and unicode in some shells/ffmpeg builds
    return f"subtitles='{p}'"

# ---------------------- route ----------------------

@app.get("/api/download/<clip_id>")
def api_download_clip(clip_id):
    """
    Download the clip, generate time-synced subtitles via Whisper (or build from provided text),
    optionally translate into `lang` query param, burn subtitles, and return subtitled MP4.

    Query params:
      lang=<iso>   e.g. lang=es or lang=fr   -> translates SRT into this language
      text=<urlencoded text> -> optional: provide narration text so we can build SRT from text instead of whisper
    """
    info = get_clip(clip_id)
    if info.get("status") != "done":
        return jsonify({"ok": False, "error": "clip not ready"}), 409

    url = info.get("result_url")
    if not url:
        return jsonify({"ok": False, "error": "missing result_url"}), 500

    target_lang = request.args.get("lang")     # e.g. "es"
    provided_text = request.args.get("text")   # if provided, make SRT from this text

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{clip_id}_{ts}.mp4"
    dest = DOWNLOAD_DIR / name

    # 1) Download MP4
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        app.logger.error("Failed downloading clip: %s", e)
        return jsonify({"ok": False, "error": "download failed"}), 502

    # prepare srt path
    srt_path = dest.with_suffix(".srt")

    # 2) If provided text, build a simple SRT from it (no timestamps accuracy)
    if provided_text:
        srt_text = make_srt_from_text(provided_text, seconds_per_line=3)
        srt_path.write_text(srt_text, encoding="utf-8")
    else:
        # 3) Extract audio and run Whisper transcription -> SRT
        wav_path = dest.with_suffix(".wav")
        try:
            run(["ffmpeg", "-y", "-i", str(dest), "-ac", "1", "-ar", "16000", str(wav_path)], check=True)
        except CalledProcessError as e:
            app.logger.error("ffmpeg audio extraction failed: %s", e)
            return send_file(dest, as_attachment=True, download_name=name, mimetype="video/mp4")

        try:
            # note: make_srt_with_whisper should write the srt_path; you pass language=None to autodetect
            make_srt_with_whisper(wav_path, srt_path, model_size="small", language=None)
        except Exception as e:
            app.logger.error("[subtitles] Whisper failed: %s", e)
            # fallback: return original video if transcription fails
            return send_file(dest, as_attachment=True, download_name=name, mimetype="video/mp4")

    # 4) If translation requested, translate SRT preserving timestamps
    final_srt_path = srt_path
    if target_lang:
        raw = srt_path.read_text(encoding="utf-8")
        items = parse_srt_blocks(raw)
        if not items:
            app.logger.error("No SRT items parsed; returning original video")
            return send_file(dest, as_attachment=True, download_name=name, mimetype="video/mp4")
        try:
            translated_items = translate_srt_items(items, target_lang)
            translated_text = build_srt_from_items(translated_items)
            final_srt_path = dest.with_name(dest.stem + f"_{target_lang}.srt")
            final_srt_path.write_text(translated_text, encoding="utf-8")
        except Exception as e:
            app.logger.error("Translation step failed: %s -- falling back to original SRT", e)
            final_srt_path = srt_path

    # 5) Burn subtitles using ffmpeg (ensure we resolve and escape path)
    subtitled = dest.with_name(dest.stem + (f"_subt_{target_lang or 'orig'}.mp4"))
    try:
        srt_for_filter = ffmpeg_subtitles_filter_arg(final_srt_path)
        run([
            "ffmpeg", "-y",
            "-i", str(dest),
            "-vf", srt_for_filter,
            "-c:a", "copy",
            str(subtitled)
        ], check=True)
    except CalledProcessError as e:
        app.logger.error("ffmpeg subtitle burn failed: %s", e)
        # fallback: return original video
        return send_file(dest, as_attachment=True, download_name=name, mimetype="video/mp4")

    # 6) Clean small artifacts (wav + maybe original srt) if you want
    try:
        wavp = dest.with_suffix(".wav")
        if wavp.exists():
            wavp.unlink()
    except Exception:
        pass

    return send_file(subtitled, as_attachment=True, download_name=subtitled.name, mimetype="video/mp4")



# --- Math Tutor (Gemini) ---
@app.post("/api/math-tutor")
def api_math_tutor():
    """
    Body: { "question": "...", "topic": "..." }
    Returns: { ok, speech, steps[], latex, answer }
    """
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    topic    = (payload.get("topic") or "").strip()

    if not question:
        return jsonify({"ok": False, "error": "question required"}), 400

    # Ask Gemini to return STRICT JSON
    system = (
        "You are a helpful math tutor. "
        "Return STRICT JSON with these keys only: "
        "speech (string, one friendly sentence), "
        "steps (array of short strings), "
        "latex (LaTeX string for the key expression/derivation), "
        "answer (short final result). "
        "Do not include any extra commentary. No markdown, no code fences."
    )
    prompt = f"{system}\n\nQuestion: {question}\nTopic (optional): {topic}"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Strong hint so Gemini serializes JSON without prose
        "generationConfig": {"response_mime_type": "application/json"},
    }

    try:
        r = requests.post(GEMINI_URL, headers=headers, json=body, timeout=60)
    except requests.RequestException as e:
        return jsonify({"ok": False, "error": f"upstream error: {e}"}), 502

    if r.status_code != 200:
        return jsonify({"ok": False, "error": r.text}), 502

    # Extract model text safely
    try:
        data = r.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return jsonify({"ok": False, "error": "bad gemini response"}), 502

    # Some models still wrap with ```json ... ```
    def strip_fences(t: str) -> str:
        t = t.strip()
        if t.startswith("```"):
            # remove first line ```
            t = t.split("```", 1)[-1]
            # remove trailing fences if present
            if t.strip().endswith("```"):
                t = t.rsplit("```", 1)[0]
        # Remove leading "json" label lines if any
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
        return t.strip()

    raw_clean = strip_fences(raw)

    # Parse JSON or fallback
    try:
        parsed = json.loads(raw_clean)
    except Exception:
        # very defensive minimal fallback
        parsed = {
            "speech": raw.strip()[:180],
            "steps": [],
            "latex": "",
            "answer": ""
        }

    # Normalize shapes
    speech = (parsed.get("speech") or "").strip()
    steps  = parsed.get("steps") or []
    if isinstance(steps, str):
      steps = [steps]
    latex  = (parsed.get("latex") or "").strip()
    answer = (parsed.get("answer") or "").strip()

    return jsonify({
        "ok": True,
        "speech": speech,
        "steps": steps,
        "latex": latex,
        "answer": answer
    })

@app.post("/api/translate")
def api_translate():
    """
    Body: { "text": "...", "target_language": "Kannada" }
    Returns: { ok, translation }
    """
    payload = request.get_json(force=True) or {}
    text = (payload.get("text") or "").strip()
    target_lang = (payload.get("target_language") or "English").strip()

    if not text:
        return jsonify({"ok": False, "error": "text required"}), 400

    prompt = (
        f"Translate the following text into {target_lang}. "
        "Return the translation string ONLY. Do not wrap in quotes or markdown. "
        "Keep the tone natural and conversational.\n\n"
        f"Text to translate: {text}"
    )

    try:
        translation = call_gemini(prompt).strip()
        # Clean up common Gemini wrapping
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]
        return jsonify({"ok": True, "translation": translation})
    except Exception as e:
        app.logger.error(f"Translation Error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# --- in app.py ---
import base64, os, uuid, json
from pathlib import Path
from flask import request, jsonify

GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")
AUDIO_OUT = Path("static/audio")
AUDIO_OUT.mkdir(parents=True, exist_ok=True)

def build_ssml(script_segments):
    """
    script_segments: list of { id: 'S1', text: 'Say this...' }
    Returns SSML string with <mark name="..."/> before each segment.
    """
    # keep it simple: each line is a paragraph for better prosody
    parts = []
    parts.append('<speak>')
    for seg in script_segments:
        safe_text = seg["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts.append(f'<p><mark name="{seg["id"]}"/>{safe_text}</p>')
    parts.append('</speak>')
    return "".join(parts)

@app.post("/api/tts-lesson")
def api_tts_lesson():
    """
    Body:
    {
      "voice": {"languageCode":"en-US","name":"en-US-Neural2-C"},
      "segments": [{ "id":"S1", "text":"Let's solve 9x - 2 = 0." }, ...]
    }
    Returns: { ok, audio_url, marks:[{name, timeMs}], sampleRateHz }
    """
    if not GOOGLE_TTS_API_KEY:
        return jsonify({"ok": False, "error": "Missing GOOGLE_TTS_API_KEY"}), 500

    data = request.get_json(force=True) or {}
    segments = data.get("segments") or []
    if not segments:
        return jsonify({"ok": False, "error": "segments required"}), 400

    voice = data.get("voice") or {"languageCode":"en-US","name":"en-US-Neural2-C"}
    ssml = build_ssml(segments)

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
    body = {
        "input": {"ssml": ssml},
        "voice": voice,
        "audioConfig": {
            "audioEncoding": "MP3",
            # You can set speakingRate, pitch, effectsProfileId, etc.
        },
        # Ask for SSML mark timepoints
        "enableTimePointing": ["SSML_MARK"]
    }

    import requests
    r = requests.post(url, json=body, timeout=60)
    if r.status_code != 200:
        return jsonify({"ok": False, "error": r.text}), 502

    try:
        payload = r.json()
        audio_b64 = payload["audioContent"]
        timepoints = payload.get("timepoints", [])  # [{markName, timeSeconds}, ...]
    except Exception:
        return jsonify({"ok": False, "error": "bad TTS response"}), 502

    # save audio
    fname = f"{uuid.uuid4().hex}.mp3"
    fpath = AUDIO_OUT / fname
    with open(fpath, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    marks = [{"name": tp["markName"], "timeMs": int(float(tp["timeSeconds"]) * 1000)} for tp in timepoints]
    return jsonify({
        "ok": True,
        "audio_url": f"/static/audio/{fname}",
        "marks": marks
    })


from requests.auth import HTTPBasicAuth

def did_auth():
    # DID_API_KEY is loaded earlier
    key = os.environ.get("DID_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DID_API_KEY not set")
    return HTTPBasicAuth(key, "")

DID_API_BASE = "https://api.d-id.com"



@app.get("/api/agent")
def api_agent():
    """
    Fetch the user's custom D-ID agent (avatar) from /agents/me.
    Returns JSON: { ok: True, agent: {...} } or { ok: False, error: "..." }.
    """
    key = os.environ.get("DID_API_KEY", "").strip()
    if not key:
        app.logger.error("DID_API_KEY not set in environment")
        return jsonify({
            "ok": False,
            "error": "Server misconfiguration: missing DID_API_KEY"
        }), 500

    url = "https://api.d-id.com/agents/me"
    headers = {"Accept": "application/json"}

    try:
        r = requests.get(url, auth=HTTPBasicAuth(key, ""), headers=headers, timeout=30)
        app.logger.info("D-ID /agents/me status=%s", r.status_code)
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        app.logger.warning("D-ID API error /agents/me: %s %s", r.status_code if 'r' in locals() else '??', getattr(r, "text", ""))
        return jsonify({
            "ok": False,
            "error": f"D-ID API error {getattr(r, 'status_code', '?')}: {getattr(r, 'text', '')}"
        }), 502
    except Exception as e:
        app.logger.exception("Failed to reach D-ID /agents/me")
        return jsonify({
            "ok": False,
            "error": f"Failed to fetch agent: {str(e)}"
        }), 502

    # Try to parse JSON safely
    try:
        data = r.json()
    except Exception:
        app.logger.warning("D-ID returned non-JSON for /agents/me: %s", r.text[:300])
        return jsonify({
            "ok": False,
            "error": "Invalid JSON response from D-ID"
        }), 502

    # Return success
    return jsonify({"ok": True, "agent": data}), 200




@app.get("/api/user-avatars")
def api_user_avatars():
    try:
        auth = did_auth()
        # Fetch from /scenes/avatars
        r = requests.get(f"{BASE}/scenes/avatars", auth=auth)
        if r.status_code == 200:
            data = r.json()
            # Extract avatars list
            avatars = data.get("avatars", [])
            return jsonify({"ok": True, "avatars": avatars})
        else:
            return jsonify({"ok": False, "error": r.text}), r.status_code
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/")
def health():
    return jsonify({"status": "Teachly backend live 🚀"})


