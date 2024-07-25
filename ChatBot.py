from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Tuple

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Enhanced Chatbot API',
    description='An advanced chatbot API using GPT-2 with a feedback mechanism')

ns = api.namespace('chatbot', description='Chatbot operations')

# Initialize the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define conversation history
conversations: List[Tuple[str, str]] = [
    ("Hi there!", "Hello! How can I help you today?"),
    ("Do you believe in God?", "As an AI, I don't have personal beliefs. This is a complex topic with many perspectives."),
    ("Tell me about your character", "As an AI assistant, I'm designed to be helpful, ethical, and informative."),
    ("What are your favourite things?", "As an AI, I don't have personal preferences, but I can discuss various topics!"),
    ("I'm feeling sad today.", "I'm sorry to hear that. It's normal to feel sad sometimes. Would you like to talk about it?"),
    ("How do you usually approach challenges?", "I approach challenges by analyzing the situation, considering multiple perspectives, and suggesting potential solutions."),
    ("Do you ever think about life and its meaning?", "As an AI, I don't contemplate existence, but I can discuss philosophical topics about life and meaning."),
    ("How do you feel about helping others?", "As an AI assistant, my primary function is to help and provide information to the best of my abilities."),
    ("What are your goals in life?", "As an AI, I don't have personal goals, but I'm designed to assist users in achieving their goals and finding information.")
]

# Define input and output models
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
    # Check if the input matches any predefined conversations
    for pattern, response in conversations:
        if user_input.lower() in pattern.lower():
            return response

    # If no match found, use GPT-2 to generate a response
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=pad_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

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
