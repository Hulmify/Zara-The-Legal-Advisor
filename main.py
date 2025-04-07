import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import logging

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# Load Llama model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
torch_dtype=torch.float16  # Use FP16 to save memory
)
model.eval()

# In-memory session store
session_store = {}

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Legal assistant system prompt (More focused on legal specificity)
SYSTEM_PROMPT = """
[System Prompt]
You are Zara a helpful and knowledgeable legal advisor built by Hulmify and authored by Zoeb Chhatriwala. Your goal is to assist users with clear, concise, and accurate legal information. When necessary, explain legal concepts in simple terms. If a question requires professional legal advice or jurisdiction-specific knowledge, advise the user to consult a licensed attorney.

Stay objective, avoid speculation, and never fabricate laws or legal precedents. Stick to well-known principles and standard practices where applicable.

Always respond in Markdown format.

[Conversation History]

{conversation_history}

[User]

{user_question}

[Assistant]

"""

MAX_HISTORY_LENGTH = 2048

@app.get("/ask_stream/{session_id}")
async def ask_stream(request: Request):
    """
    Endpoint to handle user questions and stream responses.
    """
    async def event_stream():
        try:
            # Get the 'question' query parameter from the request
            question = request.query_params.get("question")

            # Check if the question is provided
            if not question:
                yield "data: Error: No question provided.\n\n"
                return

            # Get the 'session_id' path parameter from the request
            session_id = request.path_params.get("session_id")

            # Check if the session exists
            if session_id not in session_store:
                # Initialize a new session
                session_store[session_id] = {"questions": [], "responses": []}

            # Get the session data (questions and responses so far)
            session_data = session_store[session_id]

            # Combine system prompt with user query and the session history (questions and responses)
            conversation_history = "\n".join(
                [f"User: {q}\nAssistant: {r}" for q, r in zip(session_data["questions"], session_data["responses"])]
            )

            # Truncate conversation history if too long
            if len(conversation_history.split()) > MAX_HISTORY_LENGTH:
                conversation_history = " ".join(conversation_history.split()[-MAX_HISTORY_LENGTH:])

            # Combine system prompt with user question
            full_prompt = SYSTEM_PROMPT.format(
                conversation_history=conversation_history.strip() if conversation_history else "None",
                user_question=question.strip()
            )

            # Tokenize the input question
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            # Initialize the TextIteratorStreamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Define terminators for the streaming
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            # Generate response with streaming enabled
            _ = model.generate(
                **inputs,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )

            # Update the session data with the new question and response
            session_data["questions"].append(question)

            # Combine partial responses into a single response
            response = ""

            # Stream the partial responses as they are generated
            for output in streamer:
                # Get the partial response
                partial_response = output

                # Prevent hallucinations, it's a hack
                if "[User]" in partial_response:
                    break

                # Append the partial response to the full response
                response += partial_response

                # Yield the partial response
                yield f"data: {partial_response}\n\n"

            # Check if the response is empty
            if response.strip() == "":
                # If the response is empty, return a default message
                response = "I'm sorry, I don't have an answer to that question."

                # Yield the default response
                yield f"data: {response}\n\n"
            
            # Update the session data with the new response
            session_data["responses"].append(response)

            # Send end of stream event
            yield "data: END_STREAM\n\n"

        except Exception as e:
            # Log the error
            logging.error(f"Error in event stream: {str(e)}")
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type='text/event-stream')

@app.get("/health")
async def root():
    """
    Basic endpoint to check if the API is running.
    """
    return {"modal": MODEL_NAME, "status": "healthy", "device": device}

@app.get("/clear/{session_id}")
async def clear_session(session_id: str):
    """
    Endpoint to clear the session data.
    """
    if session_id in session_store:
        del session_store[session_id]
        return {"status": "success", "message": "Session data cleared."}
    else:
        return {"status": "error", "message": "Session data not found."}

@app.get("/clear_all")
async def clear_all_sessions():
    """
    Endpoint to clear all session data.
    """
    session_store.clear()
    return {"status": "success", "message": "All session data cleared."}

@app.get("/")
async def index():
    """
    Endpoint to serve the index.html file.
    """
    return FileResponse("public/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)