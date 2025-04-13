import pandas as pd
import openai
import faiss
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your FAQ Excel file
def load_faqs(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df.dropna(subset=["User Query", "Product Responses"], inplace=True)
    df["text"] = df["User Query"] + " " + df["Product Responses"]
    return df


# Get embeddings using OpenAI
def get_embeddings(texts, model="text-embedding-3-small"):
    embeddings = []
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

    return np.array(embeddings).astype("float32")

# Save FAISS index
def save_index(index, index_path, df):
    faiss.write_index(index, index_path)
    df.to_json("vector_store/faq_metadata.json", orient="records")

# Main function
def main():
    os.makedirs("vector_store", exist_ok=True)

    df = load_faqs("/Users/sravangunapati/Desktop/Projects/Personal/dataset/faqs.xlsx")

    embeddings = get_embeddings(df["text"].tolist())

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    save_index(index, "vector_store/faq_index.faiss", df)


if __name__ == "__main__":
    main()
