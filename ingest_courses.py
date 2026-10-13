import os
import torch
import chromadb
import pandas as pd
import requests
import io
from sentence_transformers import SentenceTransformer
import gc


# --- CONFIGURATION ---
DB_DIR = "./vector_db"
COLLECTION_NAME = "learning_courses"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class RAGIngestor:
    def __init__(self):
        print("⚙️ Initializing local Embedding Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using Device: {self.device.upper()}")
        
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        print("🗄️ Initializing Local ChromaDB Client...")
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print("✅ Core systems initialized.")

    def clear_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def process_and_ingest(self):
        print("🤗 Downloading dataset via standard HTTPS download link...")
        
        # Direct download endpoint from Hugging Face CDN
        CSV_URL = "https://huggingface.co/datasets/azrai99/coursera-course-dataset/resolve/main/coursera_course_2024.csv"
        
        try:
            # Send standard requests call to stream the binary data payload
            response = requests.get(CSV_URL)
            response.raise_for_status()
            
            # Read the CSV directly out of the memory buffer stream
            df = pd.read_csv(io.StringIO(response.text))
            print(f"📖 Dataset successfully fetched! Loaded {len(df)} course records.")
        except Exception as e:
            print(f"❌ Error downloading file directly: {e}")
            return
        
        # Fill empty fields to ensure parsing doesn't crash text assembly
        df['title'] = df['title'].fillna("Unknown Course")
        df['Description'] = df['Description'].fillna("")
        df['Skills'] = df['Skills'].fillna("General Learning")
        
        documents = []
        metadatas = []
        ids = []

        print("🧠 Building vector text contexts and structuring metadata...")
        for idx, row in df.iterrows():
            title = str(row['title'])
            description = str(row['Description'])
            skills = str(row['Skills'])
            
            # Combine text fields to give our embedding vectors dense semantic context
            combined_text = f"Course: {title}. Skills taught: {skills}. Summary: {description}"
            
            documents.append(combined_text)
            ids.append(f"course_{idx}")
            metadatas.append({
                "title": title,
                "skills": skills
            })

        print(f"⚡ Generating Vector Embeddings for {len(documents)} courses on GPU...")
        # Local PyTorch model encodes documents into mathematical coordinate tensors
        embeddings = self.model.encode(documents, convert_to_tensor=False, show_progress_bar=True)
        
        print("📥 Storing vector documents into ChromaDB local index...")
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Batch insert to keep memory allocation completely safe on your 6GB VRAM hardware
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                embeddings=embeddings_list[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        
        print(f"🎉 Success! Ingested database ready at local directory '{DB_DIR}'.")
        self.clear_vram()

if __name__ == "__main__":
    ingestor = RAGIngestor()
    ingestor.process_and_ingest()