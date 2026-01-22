# Whiteboard Logic — Intelligent Generation + Structured Parsing + Audio-Visual Sync (Interview Explanation)

## What it is (1–2 lines)
**Whiteboard Logic** is a teaching-style output engine that transforms an AI response into a **step-by-step whiteboard animation** synced with an avatar video.  
It bridges **unstructured AI reasoning** and a **structured, reliable UI flow** using a lightweight **DSL (Domain Specific Language)**.

---

## Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    U[User Topic / Question] --> FE[Frontend UI]
    FE -->|POST /api/generate-whiteboard-script| BE[Flask Backend]

    BE -->|System Prompt + DSL tags| LLM[Gemini 2.0\nWhiteboard Engine]
    LLM -->|Tagged Script\nSTEP_n / DIAGRAM / HIGHLIGHTS| BE
    BE --> FE

    FE --> PARSER[JS Parser\nRegex Tag Extraction]
    PARSER --> STATE[State Objects Array\n{step, diagram, highlights}]
    STATE --> RENDER[Whiteboard Renderer\nStep Cards + Dot Grid + Kalam Font]

    FE --> DID[D-ID Avatar Video]
    DID --> VIDEO[HTML5 Video Player]

    VIDEO -->|ontimeupdate| SYNC[Sync Engine\nChar-Ratio Thresholds]
    SYNC -->|Reveal Step| RENDER
```

---

## Core Concept: The Bridge (AI → UI Animation)
> “I didn’t want the AI to just text the answer — I wanted it to teach step-by-step.”

To achieve this, I enforced a **structured output format** using custom tags like:
- `STEP_1:`
- `STEP_2:`
- `DIAGRAM:`
- `HIGHLIGHTS:`

This tagged structure works like a **mini scripting language** (DSL) so the frontend can parse the answer deterministically and render it as a lesson.

---

## Architecture Overview

### Components
- **Backend (Reasoning Engine)**
  - Gemini 2.0 “Whiteboard Engine” prompting
  - API endpoint: `/api/generate-whiteboard-script`
  - Output: structured tagged script
- **Frontend (Parser + Renderer + Sync Engine)**
  - Regex-based tag extraction
  - Step cards + diagram area
  - Time-based reveal synced to video
- **Avatar Video Layer (D-ID)**
  - Produces the teaching avatar video
  - Video playback time drives UI sync

---

## High-Level Technical Flow

### A) Reasoning Engine (Backend)

#### 1) Constrained System Prompting (Gemini 2.0)
The backend uses **Google Gemini 2.0** with a strict system prompt that forces the LLM to behave as a **Whiteboard Engine**:

- Step-by-step output only
- Consistent tags for parsing
- Optional ASCII diagrams + formulas
- Clear highlights / key points per step

✅ This reduces unpredictable formatting and makes the response machine-readable.

#### 2) API Endpoint Hand-off
A Flask endpoint handles the full generation pipeline:

`POST /api/generate-whiteboard-script`
- Input: user topic/question
- Output: tagged structured response (DSL-style)

---

### B) Parser (Frontend)

#### 1) Regex Processing (Tag extraction)
On the frontend, I wrote a JavaScript parser that:
- scans the raw AI text
- regex-detects tags like `STEP_1:` `DIAGRAM:` `HIGHLIGHTS:`
- converts them into a structured array like:

- `{type: "step", title: "STEP_1", content: "..."}`
- `{type: "diagram", content: "..."}`
- `{type: "highlights", bullets: [...]}`

✅ This gives predictable UI rendering and avoids brittle string-based UI hacks.

#### 2) Dynamic DOM Injection (Whiteboard UI)
Each parsed object becomes a UI block:
- rendered as a “Step Card”
- styled like a real whiteboard using:
  - handwriting font: **Kalam**
  - dot-grid CSS background
  - light transitions for reveal effect

Result: A clean “teaching board” experience instead of a plain chat response.

---

### C) Audio-Visual Synchronization (The “Magic” Part)

#### The challenge
Because the avatar (D-ID) speaks the script, the whiteboard steps must appear **exactly at the right time**.

#### Sync Strategy: Heuristic-based Timing
Instead of relying on word-level timestamps, I used a **lightweight heuristic sync algorithm**:

1) Calculate the character-length ratio of each step:
- `step_chars / total_chars`

2) Map that ratio to video playback progress:
- use HTML5 `<video>` and `ontimeupdate`

3) Reveal steps when the video reaches a threshold:
- `current_time / duration >= step_threshold`

#### Reveal mechanism
When the correct threshold is hit:
- trigger CSS transitions:
  - `opacity` (fade in)
  - `transform` (slide/scale into place)

✅ This creates the effect that the avatar is writing/teaching live.

---

## Interview Pro-Tips (Details that Impress)

### 1) Math Normalization for correct speech
Before sending the script to TTS/avatar:
- Convert symbols into readable speech

Examples:
- `^` → “to the power of”
- `*` → “multiplied by”
- `/` → “divided by”

This prevents weird pronunciation and improves teaching quality.

---

### 2) User Control: Auto-Sync Toggle
Users can turn **Auto-Sync ON/OFF**:
- ON → board reveals automatically with video
- OFF → user clicks next/prev to learn at their own pace

This improves accessibility and learning control.

---

### 3) State-Driven UI for performance
The UI is **state-driven**, meaning:
- the board reflects only the parsed state objects
- rendering stays fast and predictable
- fewer UI bugs and better responsiveness

---

## Quick Example Flow
**User asks:** “Explain RAG”  
Backend returns:
- `STEP_1:` definition  
- `STEP_2:` retrieval step  
- `DIAGRAM:` ASCII flow  
- `HIGHLIGHTS:` key points  

Frontend parses into step cards → video plays → steps appear synced.

---

## Summary for Interviewer (Perfect Closing)
> “The whiteboard logic is a bridge between AI reasoning and visual teaching.  
By enforcing a structured DSL output and syncing it with avatar speech, the system feels like an actual teacher, not just a chatbot.”

---

## One-Line Elevator Pitch
**“It’s a DSL-driven AI teaching pipeline: generate structured steps, parse reliably, and sync step reveals with avatar video playback.”**
