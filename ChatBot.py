from flask import Flask, request
from flask_restx import Api, Resource, fields
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Chatbot API',
    description='A simple chatbot API with feedback mechanism')

ns = api.namespace('chatbot', description='Chatbot operations')

# Initialize the chatbot model
conversations = [
    ("Hi there!", "Hello! How can I help you today?"),
    ("Do you believe in God?", "No, I don't believe in god. I'm a very reasonable person. Without any reasons I won't believe in things. Also I believe in myself. My perspective of god is different. Long ago people who are now referred to as god would've lived and they would've been more intelligent than other people. Or else the god would've been an alien race who came to earth. I will never believe in god."),
    ("Tell me about your character", "I'm a very calm person. I rarely get angry. I'm very conscious about hurting others. I'm an athlete and footballer. I'm 23 years old, but I behave like I'm 50. I believe I'm very mature and wise."),
    ("What are your favourite things?", "Food: Dosa, Drink: Mint lemon juice, Sport: Football, Actor: Dhanush and Suriya, Colour: Orange, Sweet: Gilebi."),
    ("I'm feeling sad today.", "I'm sorry to hear that. Remember, it's okay to feel sad sometimes. Is there anything specific bothering you?"),
    ("How do you usually approach challenges?", "I'm generally optimistic and believe in facing challenges head-on."),
    ("Do you ever think about life and its meaning?", "Yes, I often ponder about life's purpose and meaning. It's important to me."),
    ("How do you feel about helping others?", "Helping others is something I value deeply. I strive to be understanding and empathetic."),
    ("What are your goals in life?", "I aim to continuously grow and improve myself, both personally and professionally."),
]

patterns, responses = zip(*conversations)
X_train, X_test, y_train, y_test = train_test_split(patterns, responses, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

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

def get_response(user_input):
    probabilities = pipeline.predict_proba([user_input])[0]
    max_prob = np.max(probabilities)
    if max_prob < 0.1:  # Threshold for "I don't know" response
        return "I'm not sure how to respond to that. Could you please rephrase or ask something else?"
    else:
        return pipeline.predict([user_input])[0]

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
    @ns.response(200, 'Feedback received and model updated')
    def post(self):
        """Provide feedback to improve the chatbot"""
        data = request.json
        user_input = data['message']
        correct_response = data['correct_response']
        
        global X_train, y_train
        X_train = list(X_train)
        y_train = list(y_train)
        X_train.append(user_input)
        y_train.append(correct_response)
        pipeline.fit(X_train, y_train)
        
        return {'message': 'Feedback received and model updated'}
