import os
import re
import random
import nltk
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Download NLTK data
nltk_data_path = os.path.expanduser("~/nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the dataset
def load_data():
    lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    # Create a list of conversations
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))

    return id2line, conversations_ids

# Preprocess the data
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Prepare the dataset
def prepare_dataset(id2line, conversations_ids):
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            question = id2line.get(conversation[i])
            answer = id2line.get(conversation[i + 1])
            if question and answer:  # Ensure the lines exist
                questions.append(preprocess(question))
                answers.append(preprocess(answer))
    return questions, answers

# Load and prepare dataset
id2line, conversations_ids = load_data()
questions, answers = prepare_dataset(id2line, conversations_ids)

# Convert to DataFrame
df = pd.DataFrame({'question': questions, 'answer': answers})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

# Model training
model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
model.fit(X)

# Predict response
def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_input_vector = vectorizer.transform([user_input_processed])
    distances, indices = model.kneighbors(user_input_vector)
    return df.iloc[indices[0][0]]['answer']

# Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    log_interaction(user_input, response)
    return jsonify({"response": response})

# Logging interactions
import logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

def log_interaction(user_input, response):
    logging.info(f"User Input: {user_input}, Response: {response}")

if __name__ == '__main__':
    app.run(debug=True)
