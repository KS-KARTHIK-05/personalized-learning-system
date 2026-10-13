import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import gc

# --- CONFIGURATION ---
DB_DIR = "./vector_db"
COLLECTION_NAME = "learning_courses"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class RAGSearchEngine:
    def __init__(self):
        print("⚙️ Initializing Search Core...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the exact same embedding model used during ingestion
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        
        # Connect to your persistent database directory
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        print("✅ Vector database connection active.")

    def clear_vram(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def search_courses(self, skill_gap_query, top_k=3):
        """
        Converts the user's missing skill into a vector and pulls
        the top_k closest courses from ChromaDB.
        """
        print(f"\n🔍 Searching for courses matching gap: '{skill_gap_query}'...")
        
        # 1. Vectorize the search query
        query_vector = self.model.encode(skill_gap_query, convert_to_tensor=False)
        
        # 2. Query the vector database index
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )
        
        # 3. Format and present results
        print(f"🎯 Top {top_k} Recommended Courses Found:")
        print("-" * 60)
        
        recommendations = []
        
        for i in range(len(results['ids'][0])):
            metadata_entry = results['metadatas'][0][i]
            
            # CASE-INSENSITIVE FALLBACK DEFENSER:
            title = metadata_entry.get('title', metadata_entry.get('Title', 'Unknown Course'))
            skills = metadata_entry.get('skills', metadata_entry.get('Skills', 'General Learning'))
            
            document_summary = results['documents'][0][i]
            distance = results['distances'][0][i]  
            
            print(f"📘 Course Title: {title}")
            print(f"🛠️ Skills Covered: {skills}")
            print(f"📉 Distance Score: {distance:.4f}")
            print("-" * 60)
            
            recommendations.append({
                "title": title,
                "skills": skills,
                "summary": document_summary
            })
            
        self.clear_vram()
        return recommendations

if __name__ == "__main__":
    search_engine = RAGSearchEngine()
    
    # Test query mimicking a calculated skill gap from a user profile
    test_gap = "I need to learn Python data analysis libraries like Pandas and Numpy"
    
    search_engine.search_courses(skill_gap_query=test_gap, top_k=3)