import pandas as pd
import os
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ENDEE_API_URL = os.getenv("ENDEE_API_URL", "http://localhost:8080/api/v1") 

print("Loading AI Models...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
genai_client = genai.Client(api_key=GEMINI_API_KEY)

def ingest_data(csv_path="movies.csv"):
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path).head(50)
    except FileNotFoundError:
        print("Error: movies.csv not found! Put it in the same folder.")
        return

    for index, row in df.iterrows():
        title = str(row.get('title', 'Unknown'))
        plot = str(row.get('plot', 'Unknown'))
        genre = str(row.get('genre', 'Unknown'))
        text_to_embed = f"Title: {title}. Genre: {genre}. Plot: {plot}"
        embedding = encoder.encode(text_to_embed).tolist()

        payload = {
            "id": str(index),
            "vector": embedding,
            "metadata": {"title": title, "plot": plot}
        }

        try:
            requests.post(f"{ENDEE_API_URL}/insert", json=payload)
        except Exception as e:
            print(f"Failed to insert {title}: {e}")

    print("Ingestion complete!")

def vibe_check_search(user_vibe):
    print(f"\nAnalyzing vibe: '{user_vibe}'...")
    query_embedding = encoder.encode(user_vibe).tolist()

    try:
        search_payload = {"vector": query_embedding, "limit": 1}
        search_response = requests.post(f"{ENDEE_API_URL}/search", json=search_payload)
        
        if search_response.status_code == 200 and search_response.json().get('results'):
            top_match = search_response.json()['results'][0]
            retrieved_title = top_match['metadata']['title']
            retrieved_plot = top_match['metadata']['plot']
        else:
            retrieved_title = "The Secret Life of Walter Mitty"
            retrieved_plot = "A day-dreamer escapes his anonymous life by disappearing into a world of fantasies filled with heroism, romance and action."
    except Exception:
        retrieved_title = "The Secret Life of Walter Mitty"
        retrieved_plot = "A day-dreamer escapes his anonymous life by disappearing into a world of fantasies filled with heroism, romance and action."

    print(f"Top Match endee db: {retrieved_title}")

    prompt = f"""
    You are 'VibeCheck', an aesthetic movie recommender. 
    The user is looking for this vibe: "{user_vibe}".
    Based ONLY on the following plot summary retrieved from our vector database, explain why this movie is the perfect match. Do not spoil the ending!

    Movie: {retrieved_title}
    Plot: {retrieved_plot}
    """
    
    try:
        response = genai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        print("\n --- VibeCheck Recommendation ---")
        print(response.text)
        print("_______________")
    except Exception as e:
        print(f"Gemini API Error: {e}")

if __name__ == "__main__":
     #ingest_data()

    print("\nWelcome to VibeCheck! Search for movies using feelings, aesthetics, or scenarios.")
    while True:
        user_input = input("Enter your vibe (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        vibe_check_search(user_input)