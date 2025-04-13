import openai
import faiss
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it using 'export OPENAI_API_KEY=your_key'")

index = faiss.read_index("vector_store/faq_index.faiss")
faq_metadata = pd.read_json("vector_store/faq_metadata.json")

def get_review_embedding(review_text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[review_text],
        model=model
    )
    return np.array(response.data[0].embedding).astype("float32")

def is_negative_review(review_text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that classifies app reviews as positive or negative."},
            {"role": "user", "content": f"Is this review negative?\n\nReview: \"{review_text}\""}
        ]
    )
    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer

def search_faq(review_embedding, top_k=1):
    D, I = index.search(np.array([review_embedding]), top_k)
    score = D[0][0]

    return faq_metadata.iloc[I[0][0]]

def generate_response(review_text, faq=None, is_negative=False):
    if is_negative and faq is not None:
        messages = [
            {"role": "system", "content": "You are a helpful and empathetic customer support agent for a mobile app. Do not start responses with greetings like 'Thank you' or 'Hello'. Be direct and professional."},
            {"role": "user", "content": f"User review: \"{review_text}\""},
            {"role": "user", "content": f"The following FAQ might help: \nQ: {faq['User Query']}\nA: {faq['Product Responses']}"},
            {"role": "user", "content": "Write a direct, professional response to the review above. Do not start with greetings. End the response with 'Best regards,\nTeam Zaggle\nZaggle Support Team'"}
        ]
    elif is_negative:
        messages = [
            {"role": "system", "content": "You are a helpful and empathetic support assistant. Do not start responses with greetings like 'Thank you' or 'Hello'. Be direct and professional."},
            {"role": "user", "content": f"The user left a negative review: \"{review_text}\""},
            {"role": "user", "content": "Since we don't have a matching FAQ, write a direct response asking the user to contact our support team. Support hours are Monday to Friday, 6 AM to 9 PM. Do not start with greetings. End the response with 'Best regards,\nTeam Zaggle\nZaggle Support Team'"}
        ]
    else:
        messages = [
            {"role": "system", "content": """You are a friendly customer success agent for Zaggle. Do not start responses with greetings like 'Thank you' or 'Hello'. Be direct and professional. Analyze the review content and provide relevant suggestions based on these rules:
                1. If the review mentions rewards, recognition, or brand vouchers, suggest checking out the new Google Pixel vouchers available at a special price.
                2. If the review mentions KYC or payments (but not gift cards or Kuber cards), suggest trying Kuber cards for gifting to friends and family.
                3. If the review mentions gift cards or Kuber cards, suggest exploring the new bill payments module for credit card bills, electricity bills, etc.
                4. If the review mentions expenses, expense reports, advances, or EMS (Expense Management System), suggest exploring the new filters added in Expense Reports.

                Make the suggestion feel natural and relevant to their positive experience. Keep the tone friendly and encouraging."""},
            {"role": "user", "content": f"The user left a positive review: \"{review_text}\""},
            {"role": "user", "content": "Write a direct response acknowledging their positive feedback and provide a relevant suggestion based on the review content. Do not start with greetings. End the response with 'Best regards,\nTeam Zaggle\nZaggle Support Team'"}
        ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content.strip()

def handle_small_and_failure_response(rating):
    if int(rating) <= 2:
        return "I'm sorry to hear that you had a negative experience. To address your concerns more effectively, could you please contact our support team? They're available to assist you Monday to Friday, from 6 AM to 9 PM. Thank you for your understanding, and we look forward to resolving your issue."
    elif int(rating) >= 3:
        return "Thank you so much for your positive review! We're delighted to hear that you're enjoying the app."

def respond_to_review(review_text, rating=None, response_length="large"):
    try:
        if response_length == "small":
            return handle_small_and_failure_response(rating)

        print(f"\n📨 Review: {review_text}")
        if rating:
            print(f"⭐️ Rating: {rating}")

        review_embedding = get_review_embedding(review_text) if len(review_text) > 0 else None

        if rating == 3 or rating == None:
            is_negative = is_negative_review(review_text)
        elif rating <= 2:
            is_negative = True
        elif rating >= 4:
            is_negative = False

        if is_negative and review_embedding is not None:
            faq = search_faq(review_embedding)
        else:
            faq = None

        reply = generate_response(review_text, faq, is_negative)
        return reply
    except Exception as e:
        return handle_small_and_failure_response(rating)

if __name__ == "__main__":
    while True:
        user_review = input("Enter a user review (or type 'exit'): ")
        rating = int(input("Enter a rating (1-5): "))
        if user_review.lower() == "exit":
            break
        respond_to_review(user_review, rating)
