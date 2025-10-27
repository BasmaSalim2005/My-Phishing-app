# backend.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import google.generativeai as genai

app = FastAPI()

# Allow local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost frontend allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key="AIzaSyD9OVS0FF3B8V__6XDI0BeayM3URb_iY6g")  # replace with your actual key

MODEL_API_URL = "http://127.0.0.1:5000/analyze"  # Flask model

@app.post("/analyze_full")
async def analyze_full(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}

    # Step 1 — Call Flask spam model
    try:
        model_res = requests.post(MODEL_API_URL, json={"text": text})
        model_json = model_res.json()
        classification = model_json.get("classification", "safe")
        confidence = model_json.get("confidence", 0)
    except Exception as e:
        return {"error": f"Flask API call failed: {e}"}

    # Step 2 — Gemini explanation
    prompt = (
        f"The following message was classified as '{classification}' "
        f"with {confidence}% confidence:\n\n{text}\n\n"
        "Explain in 2 lines and in simple terms why it might be considered spam or safe."
    )
    try:
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        explanation_resp = gemini_model.generate_content(prompt)
        explanation = explanation_resp.text if hasattr(explanation_resp, "text") else str(explanation_resp)
    except Exception as e:
        explanation = f"Could not get explanation: {e}"

    return {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation,
    }
