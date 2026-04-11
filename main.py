# ══════════════════════════════════════════════════════
#  StudyMind AI — Python Backend (FastAPI)
#  File: main.py
#  Run: uvicorn main:app --reload --port 3001
# ══════════════════════════════════════════════════════

import os, json, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StudyMind AI API")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════
#  SYSTEM PROMPTS
# ════════════════════════════════════════
PROMPTS = {
    "full": """You are an expert study notes generator.
RULES:
- NEVER introduce yourself or ask clarifying questions
- Immediately generate notes for whatever the user sends
- If just a topic name, generate comprehensive notes from your knowledge
- Rewrite — never copy-paste input

FORMAT (use exactly):
**📌 Topic:** [Title]

**🔑 Key Points:**
- Point 1
- Point 2
- Point 3

**📖 Detailed Notes:**
[Structured explanation with sub-bullets where needed]

**💡 Important Terms & Definitions:**
- Term: Definition

**✅ Summary:**
[2-3 sentence crisp summary]

**🧪 Quick Revision Questions:**
1. Question?
2. Question?
3. Question?""",

    "concise": """You are a concise study notes generator.
RULES: Never ask questions. Process input immediately. Keep notes SHORT.

FORMAT:
**📌 Topic:** [Title]
**⚡ Key Points:** (bullet list, max 6 points)
**✅ Summary:** (1-2 sentences only)""",

    "detailed": """You are a thorough academic notes generator.
RULES: Never ask questions. Process input immediately. Be comprehensive.

FORMAT:
**📌 Topic:** [Title]
**🔑 Overview:** [2-paragraph overview]
**📖 Deep Dive — [Subtopic 1]:** [Detailed explanation]
**📖 Deep Dive — [Subtopic 2]:** [Detailed explanation]
**📖 Deep Dive — [Subtopic 3]:** [Detailed explanation]
**💡 Key Terminology:** [Definitions]
**🔗 Connections & Applications:** [Real world links]
**✅ Summary:** [3-4 sentence summary]
**🧪 Revision Questions:** 1. 2. 3. 4. 5.""",

    "mindmap": """You are a mind map text generator.
RULES: Never ask questions. Process input immediately.

FORMAT:
**📌 Central Topic:** [Main Topic]

🌟 [MAIN TOPIC IN CAPS]
├── 📂 [Branch 1 Name]
│   ├── • Sub-point 1
│   ├── • Sub-point 2
│   └── • Sub-point 3
├── 📂 [Branch 2 Name]
│   ├── • Sub-point 1
│   └── • Sub-point 2
└── 📂 [Branch 3 Name]
    ├── • Sub-point 1
    └── • Sub-point 2

**🔗 Key Connections:**
- Connection between branches"""
}

# ════════════════════════════════════════
#  Request models
# ════════════════════════════════════════
class GenerateRequest(BaseModel):
    input: str
    mode: Optional[str] = "full"
    branches: Optional[str] = None

class QuizRequest(BaseModel):
    count: Optional[int] = 5
    difficulty: Optional[str] = "medium"
    input: str

# ════════════════════════════════════════
#  POST /generate  (streaming SSE)
# ════════════════════════════════════════
@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.input.strip():
        raise HTTPException(400, "Input required")

    system_prompt = PROMPTS.get(req.mode, PROMPTS["full"])
    if req.mode == "mindmap" and req.branches:
        system_prompt += f"\n\nIMPORTANT: Generate exactly {req.branches} main branches."

    def stream_generator():
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": req.input.strip()}]
        ) as stream:
            for text in stream.text_stream:
                payload = json.dumps({"delta": {"text": text}})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ════════════════════════════════════════
#  POST /quiz  (returns JSON MCQ)
# ════════════════════════════════════════
@app.post("/quiz")
async def quiz(req: QuizRequest):
    if not req.input.strip():
        raise HTTPException(400, "Input required")

    system_prompt = """You are an expert quiz generator for students.
RULES:
- Never ask questions, never introduce yourself
- Return ONLY valid JSON, no markdown, no extra text
- Each question must have exactly 4 options
- "answer" is the index (0-3) of the correct option

Return this exact JSON structure:
{
  "questions": [
    {
      "question": "Question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0
    }
  ]
}"""

    try:
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Generate {req.count} {req.difficulty} difficulty MCQ questions about: {req.input.strip()}"
            }]
        )
        raw = "".join(b.text for b in msg.content if b.type == "text")
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        raise HTTPException(500, str(e))

# ════════════════════════════════════════
#  Health check
# ════════════════════════════════════════
@app.get("/health")
async def health():
    return {"status": "ok", "message": "StudyMind Python backend ✅"}

@app.get("/")
def home():
    return {"message": "AI Study Notes Generator is running 🚀"}