import chromadb

# Point directly to Rajesh's extracted database folder
DB_DIR = "./vector_db"

client = chromadb.PersistentClient(path=DB_DIR)

# Print every active collection table name saved inside his files
print("📋 Rajesh's Active Chroma Collections:")
print("-" * 40)
print(client.list_collections())
print("-" * 40)