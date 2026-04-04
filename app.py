"""
Water Usage Optimizer - Backend

This module serves as the backend for the Water Usage Optimizer application.
It exposes a REST API via Flask that communicates with a LangChain Agent powered by Groq LLM.

- LangChain is used to create an agent that reasons and uses tools.
- Groq LLM (via ChatGroq) is used for fast conversational agent processing.
- Tools (@tool) are custom functions the agent can execute for specific calculations.
- ConversationBufferMemory is used to store chat context during execution.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

app = Flask(__name__)
# Enable CORS for deployment compatibility (e.g. Netlify React to Render Flask)
CORS(app)

# ====================================================
# CUSTOM TOOLS FOR LANGCHAIN AGENT
# ====================================================

@tool
def calculate_daily_water_usage(members: int) -> int:
    """Calculate the estimated daily water usage in liters based on the number of family members. Assume 135 liters per person per day."""
    return members * 135

@tool
def estimate_tank_duration(tank_capacity: int, daily_usage: int) -> float:
    """Calculate how many days the water tank will last given its capacity and daily usage."""
    if daily_usage <= 0:
        return 0.0
    return round(tank_capacity / daily_usage, 2)

@tool
def suggest_water_saving_tips(area: str) -> str:
    """Provide specific water saving tips based on the concerned area (bathroom, kitchen, garden, general)."""
    area = area.lower()
    if 'bathroom' in area:
        return "Take shorter showers, install low-flow showerheads, fix leaky faucets, and turn off the tap while brushing teeth."
    elif 'kitchen' in area:
        return "Only run the dishwasher when full, reuse water from washing vegetables, and scrape plates instead of rinsing."
    elif 'garden' in area:
        return "Use drip irrigation, water plants early in the morning or late evening, and harvest rainwater."
    else:
        return "Check for hidden leaks, replace old appliances with water-efficient models, and monitor your water meter regularly."

@tool
def estimate_water_bill(total_usage: int) -> float:
    """Estimate the monthly water bill given the total daily usage in liters."""
    # Assuming standard simple rate: 30 days * $0.002 per liter
    monthly_usage = total_usage * 30
    return round(monthly_usage * 0.002, 2)

tools = [
    calculate_daily_water_usage, 
    estimate_tank_duration, 
    suggest_water_saving_tips, 
    estimate_water_bill
]

# Set up global memory for the conversational agent
# Return messages makes it compatible with chat models
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ====================================================
# ROUTES
# ====================================================

@app.route('/', methods=['GET'])
def index():
    """Serve the frontend HTML."""
    return send_file('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to verify that the backend is running."""
    return jsonify({"status": "Backend is running!"})

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Main endpoint to process water usage data.
    Accepts JSON from frontend, uses LangChain agent with Groq to analyze,
    and returns a structured JSON response.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # Extract inputs from the frontend
        members = int(data.get('members', 1))
        tank_capacity = int(data.get('tank_capacity', 1000))
        daily_usage = int(data.get('daily_usage', 200))
        area = data.get('area', 'general')
        custom_question = data.get('question', '')

        # Initialize Groq LLM (make sure GROQ_API_KEY is in environment variables)
        llm = ChatGroq(
            temperature=0.4, 
            model_name="llama3-8b-8192" 
        )

        # Initialize Langchain Agent using memory and tools
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )

        # Build prompt for the Langchain agent
        prompt = f"""
        User Profile:
        - Family Members: {members}
        - Tank Capacity: {tank_capacity} liters
        - Estimated Daily Usage: {daily_usage} liters
        - Main Area of Concern: {area}
        
        Custom Question: {custom_question}
        
        Please act as an expert water usage optimizer. Use your tools to analyze the data if needed. 
        Then, provide personalized water optimization advice based on their profile and clearly answer their custom question.
        Make sure your response directly addresses their problem.
        """

        # Run the Langchain Agent to get the AI string response
        ai_response = agent.run(prompt)

        # Use the tools directly to get precise numerical/text results for the frontend cards
        daily_est = calculate_daily_water_usage.invoke({"members": members})
        tank_dur = estimate_tank_duration.invoke({"tank_capacity": tank_capacity, "daily_usage": daily_usage})
        saving_tips = suggest_water_saving_tips.invoke({"area": area})
        bill_est = estimate_water_bill.invoke({"total_usage": daily_usage})

        # Return formatted JSON to frontend
        return jsonify({
            "daily_estimate": f"{daily_est} liters",
            "tank_duration": f"{tank_dur} days",
            "saving_tips": saving_tips,
            "bill_estimate": f"${bill_est}",
            "ai_response": ai_response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use environment variable for port if available, default to 5000
    # host="0.0.0.0" makes it accessible externally for deployments
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
