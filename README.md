# AI Tutor: Personalized Learning Platform

AI Tutor is an advanced, personalized learning platform that leverages state-of-the-art AI models to provide interactive, multi-modal educational experiences. From real-time AI video calls to adaptive whiteboard animations, AI Tutor transforms the way students learn complex subjects.

![AI Tutor Dashboard](assets/img/logo-ct.png)

## 🚀 Key Features

- **Personalized AI Tutor (AskTeach)**: An adaptive chatbot that adjusts its teaching style, tone, and pace based on your `TeachlyUserProfile`.
- **Whiteboard Animation Engine**: Automatically generates step-by-step whiteboard visualizations, ASCII diagrams, and formulas to accompany lessons.
- **C-Code Visualizer**: A memory visualization tool that traces C/UNP code execution, showing stack frames, heap allocations, and data flow in real-time.
- **Real-Time AI Video Calls**: Experience immersive learning with AI-powered avatars using D-ID and Tavus technologies.
- **Smart PDF/PPT Extraction**: Upload your lecture materials, and AI Tutor will extract the core concepts and expand them into student-friendly scripts.
- **Interactive Simulations**: Dynamic P5.js simulations for physics and math concepts, allowing students to "feel" the data.

## 🛠 Tech Stack

- **Backend**: Python, Flask, FastAPI
- **Frontend**: Vanilla JavaScript (ES6+), HTML5, CSS3 (Custom Material Theme)
- **AI Models**: 
  - **Reasoning**: Google Gemini 2.0 Flash
  - **Local LLM Support**: Mistral (via LM Studio)
  - **Speech-to-Text**: Whisper (faster-whisper)
  - **Video/Avatar Generation**: D-ID API, Tavus API
- **Visuals**: Mermaid.js (Diagrams), P5.js (Interactions), ASCII Art (Whiteboard)

## 🚦 Getting Started

### Prerequisites

- Python 3.9+
- API Keys for:
  - Google Gemini (`GEMINI_API_KEY`)
  - D-ID (`DID_API_KEY`)
  - Groq (`GROQ_API_KEY`) - *Optional*
  - Tavus (`TAVUS_API_KEY`) - *Optional*

### Installation

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/your-username/AI-Tutor.git
   cd AI-Tutor
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   ```bash
   export GEMINI_API_KEY='your_gemini_key'
   export DID_API_KEY='your_did_key'
   ```

4. **Run the Application**:
   ```bash
   ./start_server.sh
   # Or directly:
   python main.py
   ```

## 📖 Usage

- **AskTeach**: Navigate to `/askteach` to start a chat with your AI tutor. Use the sidebar to track your progress and preferences.
- **Whiteboard**: Visit `/whiteboard` to see AI-generated lecture content and animations.
- **Code Explainer**: Go to `/code-explainer` to visualize C program memory and execution steps.
- **Real-Time Call**: Experience `/realtime` for a live interaction with an AI avatar.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---
*Built with ❤️ for the future of education.*
