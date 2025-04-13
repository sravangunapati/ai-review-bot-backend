from flask import Flask, request, jsonify
import openai, faiss, numpy as np, pandas as pd, os
from dotenv import load_dotenv
from review_responder import respond_to_review, is_negative_review, search_faq, get_review_embedding, generate_response
from flask_cors import CORS
from mongo_config import get_db
from bson import ObjectId

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

db = get_db()

@app.route('/process_review', methods=['POST'])
def process_review():
    data = request.get_json()
    review_text = data.get("review_text")
    rating = data.get("rating", None)
    response_length = data.get("response_length", None)

    if not review_text:
        return jsonify({"message": "Review text is required!"}), 400

    response = respond_to_review(review_text, int(rating) if rating else None, response_length)
    
    return jsonify({
        "review_text": review_text,
        "response": response
    })

@app.route('/save_messages', methods=['POST'])
def save_messages():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({
                'error': 'Bad request',
                'message': 'Messages array is required'
            }), 400
            
        chat_data = {
            "messages": messages,
            "timestamp": pd.Timestamp.now()
        }
        result = db.chats.insert_one(chat_data)
        
        return jsonify({
            'id': str(result.inserted_id),
            'message': 'Messages saved successfully'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to save messages'
        }), 500

@app.route('/all_reviews', methods=['GET'])
def get_all_reviews():
    try:
        reviews = list(db.chats.find().sort('timestamp', -1))
        
        formatted_reviews = []
        for review in reviews:
            user_messages = [msg for msg in review['messages'] if msg.get('role') == 'user']
            title = user_messages[1]['content']if user_messages else 'No title'
            rating = user_messages[0]['content']
            
            formatted_reviews.append({
                'id': str(review['_id']),
                'title': title,
                'timestamp': review['timestamp'],
                'rating': rating
            })
        
        return jsonify({
            'data': formatted_reviews,
            'count': len(formatted_reviews)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to fetch reviews'
        }), 500

@app.route('/chat/<id>', methods=['GET'])
def get_review_by_id(id):
    try:
        chat = db.chats.find_one({'_id': ObjectId(id)})
        
        if not chat:
            return jsonify({
                'error': 'Not found',
                'message': f'Chat with ID {id} not found'
            }), 404
        
        formatted_chat = {
            'id': str(chat['_id']),
            'messages': chat['messages'],
            'timestamp': chat['timestamp']
        }
        
        return jsonify(formatted_chat)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to fetch chat'
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
