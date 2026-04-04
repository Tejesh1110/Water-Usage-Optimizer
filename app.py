"""
Water Usage Optimizer - Backend
A simple Flask API that calculates water usage stats
and uses Groq AI for smart water conservation advice.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq

app = Flask(__name__)
CORS(app)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt for the AI assistant
SYSTEM_PROMPT = (
    "You are an AI Water Usage Optimizer assistant. "
    "Help users reduce water wastage, estimate daily water usage, "
    "and suggest smart conservation tips. "
    "Keep responses concise, friendly, and actionable. "
    "Use bullet points when listing tips."
)


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"status": "Water Usage Optimizer Backend is Running!"})


@app.route("/calculate", methods=["POST"])
def calculate():
    """
    Calculate water usage statistics.
    Expects JSON: { "people": int, "daily_usage": int, "tank_capacity": int }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        people = int(data.get("people", 1))
        daily_usage = int(data.get("daily_usage", 200))
        tank_capacity = int(data.get("tank_capacity", 1000))

        # Per person usage
        per_person = round(daily_usage / people, 1) if people > 0 else 0

        # Tank duration
        tank_duration = round(tank_capacity / daily_usage, 1) if daily_usage > 0 else 0

        # Usage category
        if daily_usage <= 300:
            category = "Low Usage"
            emoji = "🟢"
            tip = "Great job! Your water usage is efficient."
        elif daily_usage <= 700:
            category = "Moderate Usage"
            emoji = "🟡"
            tip = "You're doing okay, but there's room to save more water."
        else:
            category = "High Usage"
            emoji = "🔴"
            tip = "Consider reducing your water consumption significantly."

        return jsonify({
            "people": people,
            "daily_usage": daily_usage,
            "per_person": per_person,
            "tank_capacity": tank_capacity,
            "tank_duration": tank_duration,
            "category": category,
            "emoji": emoji,
            "tip": tip,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    AI Chat endpoint using Groq.
    Expects JSON: { "message": "user question" }
    """
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "No message provided"}), 400

    try:
        user_message = data["message"]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
