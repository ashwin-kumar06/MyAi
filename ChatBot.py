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

# -------- Conversation matching utilities --------
def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

def _tokenize(text: str) -> set:
    return set(_normalize_text(text).split())

def _jaccard_similarity(a: str, b: str) -> float:
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / max(1, len(union))

def _find_best_conversation_match(user_input: str) -> Tuple[str, float] | Tuple[None, float]:
    best_response = None
    best_score = 0.0
    normalized_user = _normalize_text(user_input)

    for pattern, response in conversations:
        normalized_pattern = _normalize_text(pattern)

        # Exact match
        if normalized_user == normalized_pattern:
            return response, 1.0

        # Substring containment (either direction) gets a strong score
        if normalized_user in normalized_pattern or normalized_pattern in normalized_user:
            score = 0.85
        else:
            score = _jaccard_similarity(normalized_user, normalized_pattern)

        if score > best_score:
            best_score = score
            best_response = response

    return (best_response, best_score) if best_response is not None else (None, 0.0)

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
    # Always check stored conversations first using fuzzy matching
    matched_response, score = _find_best_conversation_match(user_input)
    if matched_response is not None and score >= 0.45:
        return matched_response

    # Otherwise query the model, but include the stored conversations as few-shot guidance
    try:
        example_messages: List[dict] = []
        # Limit examples to avoid overly long prompts
        max_pairs = 20
        for i, (q, a) in enumerate(conversations[:max_pairs]):
            example_messages.append({"role": "user", "content": q})
            example_messages.append({"role": "assistant", "content": a})

        system_instruction = (
            "You are Ashwin’s assistant. First, try to answer using the provided prior "
            "Q&A examples verbatim if they match the user's question. If none match, "
            "answer concisely and helpfully, staying consistent with those examples."
        )

        messages = (
            [{"role": "system", "content": system_instruction}] +
            example_messages +
            [{"role": "user", "content": user_input}]
        )

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
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
