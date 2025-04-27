import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import difflib
from openai import AsyncOpenAI
from fastapi.staticfiles import StaticFiles

# Set up OpenAI client with API key from environment
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CodeModificationRequest(BaseModel):
    code: str
    prompt: str

# Initialize FastAPI app
app = FastAPI(title="Code Modifier API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def serve_editor():
    """Serve the main editor page."""
    return FileResponse("frontend/index.html")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend", html=False), name="static")

@app.post("/modify")
async def modify_code(request: CodeModificationRequest):
    """
    Process code modification requests using OpenAI's API.
    Returns explanation, new code, and diff of changes.
    """
    response = await openai_client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        # model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant specialized in game development. "
                    "Your job is to help game developers improve their code snippets by making modifications based on user instructions.\n\n"
                    "Always reply with:\n"
                    "### Explanation:\n- Short bullet points explaining what you did and why.\n\n"
                    "### Code:\n"
                    "-The full updated code snippet.\n\n"
                    "Focus on clean, optimized, and game-performance-friendly code. "
                    "Preserve any game logic unless the user asks to modify it."
                )
            },
            {
                "role": "user",
                "content": f"### Code:\n{request.code}\n\n### Change request:\n{request.prompt}"
            }
        ],
        temperature=0.2
    )

    response_text = response.choices[0].message.content or ""
    
    # Parse response based on format markers
    if "### Code:" in response_text:
        explanation, _, code_section = response_text.partition("### Code:")
        explanation = explanation.replace("### Explanation:", "").strip()
        modified_code = code_section.strip()
    else:
        explanation = response_text.strip()
        modified_code = response_text.strip()

    # Generate diff between original and modified code
    code_diff = "\n".join(difflib.unified_diff(
        request.code.splitlines(),
        modified_code.splitlines(),
        fromfile="original", tofile="modified",
        lineterm=""
    ))

    return {
        "explanation": explanation,
        "new_code": modified_code,
        "diff": code_diff
    }
