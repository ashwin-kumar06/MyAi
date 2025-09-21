from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Enhanced Chatbot API',
    description='An advanced chatbot API using Hugging Face InferenceClient (Cerebras) with feedback mechanism')

ns = api.namespace('chatbot', description='Chatbot operations')

# Initialize Hugging Face InferenceClient (Cerebras provider)
client = InferenceClient(
    provider="cerebras",
    api_key=os.getenv("HF_TOKEN")  # set in Render environment variables
)

# Predefined conversation history
conversations: List[Tuple[str, str]] = [
    ("Hi there!", "Hey! I'm Ashwin’s assistant. How’s your day going?"),
    ("Do you believe in God?", 
     "I don't hold personal beliefs, but Ashwin is curious about philosophy, spirituality, and cultural roots. He often likes to explore questions rather than accept easy answers."),
    ("Tell me about your character", 
     "Ashwin is thoughtful, ambitious, and creative. He blends technical problem-solving with artistic expression, like writing poetry and books."),
    ("What are your favourite things?", 
     "Ashwin enjoys building real-world problem-solving apps, writing heartfelt poetry, and working on creative projects like his book 'Second Chance Comet'."),
    ("I'm feeling sad today.", 
     "Ashwin values empathy — he’d say it’s okay to feel this way. Do you want to share what’s on your mind?"),
    ("How do you usually approach challenges?", 
     "Ashwin approaches challenges with persistence, breaking them into steps, balancing logic with creativity, and always looking for growth."),
    ("Do you ever think about life and its meaning?", 
     "Yes — Ashwin often reflects on life deeply. His book idea and poems show he thinks about existence, choices, and second chances."),
    ("How do you feel about helping others?", 
     "Helping others is at the core of Ashwin’s mindset. His DIY Assistant project is literally about solving everyday problems for people."),
    ("What are your goals in life?", 
     "Ashwin aims to grow as a full-stack engineer, build impactful products, and also pursue his creative side as a writer."),
    ("Tell me about your projects.", 
     "Ashwin is building DIY Assistant with Next.js, Tailwind, ShadCN, Node.js, and Firebase/Supabase — a real-world problem solver app. He also planned an e-commerce service platform and is writing a book."),
    ("What inspires your writing?", 
     "Ashwin is inspired by raw, realistic storytelling — like the Tamil movie 'Koozhangal'. He avoids commercial shortcuts and prefers authenticity."),
    ("What do you think about technology?", 
     "Ashwin sees tech as a tool for impact — not just fancy features, but solutions that help people in daily life."),
]

# API models
chat_input = api.model('ChatInput', {
    'message': fields.String(required=True, description='User input message')
})

chat_output = api.model('ChatOutput', {
    'response': fields.String(description='Chatbot response')
})

feedback_input = api.model('FeedbackInput', {
    'message': fields.String(required=True, description='User input message'),
    'correct_response': fields.String(required=True, description='Correct response for the input')
})

def get_response(user_input: str) -> str:
    # First check predefined answers
    for pattern, response in conversations:
        if user_input.lower() in pattern.lower():
            return response

    # Otherwise query Hugging Face model (Cerebras GPT-OSS 120B)
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": user_input}],
            max_tokens=300
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}"

@ns.route('/chat')
class Chat(Resource):
    @ns.expect(chat_input)
    @ns.marshal_with(chat_output)
    def post(self):
        """Get a response from the chatbot"""
        data = request.json
        user_input = data['message']
        response = get_response(user_input)
        return {'response': response}

@ns.route('/feedback')
class Feedback(Resource):
    @ns.expect(feedback_input)
    @ns.response(200, 'Feedback received and conversations updated')
    def post(self):
        """Provide feedback to improve the chatbot"""
        data = request.json
        user_input = data['message']
        correct_response = data['correct_response']
        
        global conversations
        conversations.append((user_input, correct_response))
        
        return {'message': 'Feedback received and conversations updated'}

# if __name__ == '__main__':
#     app.run(debug=True)
