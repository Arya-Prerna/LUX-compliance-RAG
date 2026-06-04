import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Load environment variables (.env)
load_dotenv()
print(f"Debug: API Key found? {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")

class LuxComplianceRAG:
    def __init__(self):
        # Initialize Groq Client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize Qdrant Client
        self.qdrant_client = QdrantClient(path="data/qdrant_db")
        
        # Initialize Embedding Model
        print("Loading Embedding Model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        self.collection_name = "lux_circulars"

    def get_context(self, query, limit=3):
        """Search Qdrant for relevant chunks."""
        query_vector = self.model.encode(query).tolist()
        
        # Using the query_points method that worked in our test
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        context_list = []
        for res in response.points:
            context_list.append({
                "text": res.payload.get("text"),
                "source": res.payload.get("source"),
                "page": res.payload.get("page")
            })
        return context_list

    def ask(self, question):
        """Full RAG Pipeline: Retrieve -> Prompt -> Generate."""
        # 1. Retrieval
        print(f"\nSearching for context...")
        context_chunks = self.get_context(question)
        
        if not context_chunks:
            return "I couldn't find any relevant Luxembourgish regulations to answer that."

        # 2. Build the Context String
        formatted_context = ""
        sources = []
        for i, c in enumerate(context_chunks):
            formatted_context += f"--- CONTEXT {i+1} (Source: {c['source']}, Page: {c['page']}) ---\n{c['text']}\n\n"
            sources.append(f"{c['source']} (Page {c['page']})")

        # 3. Create the System Prompt (The "Brain" part)
        system_prompt = """
        You are a specialized Compliance AI assistant for the Luxembourg financial sector.
        You will be provided with snippets from CSSF circulars and EU regulations.
        
        Your task:
        1. Answer the user's question accurately using ONLY the provided context.
        2. If the context is in French or German, translate the core meaning into English.
        3. ALWAYS mention which document and page number you are referring to.
        4. If you cannot find the answer in the context, say you don't know.
        """

        # 4. Generate Answer using Groq
        print(f"Generating answer using Llama-3...")
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {question}"}
            ],
            model="llama-3.3-70b-versatile", # One of the most powerful open models
            temperature=0.2, # Keep it professional and focused
        )

        answer = chat_completion.choices[0].message.content
        return answer, list(set(sources))

# --- Main Interface ---
if __name__ == "__main__":
    rag = LuxComplianceRAG()
    
    print("\n" + "="*50)
    print("WELCOME TO LUX-COMPLIANCE RAG")
    print("="*50)
    
    while True:
        user_query = input("\nAsk a question about Lux Regulations (or type 'quit'): ")
        if user_query.lower() == 'quit':
            break
            
        answer, sources = rag.ask(user_query)
        
        print("\n" + "-"*30 + " ANSWER " + "-"*30)
        print(answer)
        print("\nSources used:", ", ".join(sources))
        print("-" * 68)