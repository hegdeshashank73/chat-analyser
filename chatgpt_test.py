import weaviate
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class RAGSystem:
    def __init__(
        self,
        openai_api_key: str,
        class_name: str,
        embedding_field: str = "content"
    ):
        self.client = weaviate.connect_to_local()
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.collections.get(class_name)
        self.embedding_field = embedding_field

    def is_informative_content(self, text: str, query: str) -> bool:
        """
        Filter out non-informative or circular content
        """
        # Clean the text and query for comparison
        text_lower = text.lower()
        query_lower = query.lower().rstrip('?')
        
        # Remove the query text from the chunk
        cleaned_text = text_lower.replace(query_lower, "").strip()
        
        # Basic checks for non-informative content
        if len(cleaned_text.split()) < 5:  # Too short after removing query
            return False
            
        # Check if it's just a question about the topic
        if text_lower.startswith(query_lower) and '?' in text:
            return False
            
        # Check if it's just a reference without explanation
        reference_phrases = ["like", "similar to", "same as"]
        if any(phrase in cleaned_text and len(cleaned_text.split()) < 10 for phrase in reference_phrases):
            return False
            
        return True

    def retrieve_relevant_context(
        self,
        query: str,
        limit: int = 7,
        distance_threshold: float = 0.75
    ) -> List[Dict[Any, Any]]:
        """
        Retrieve and filter relevant documents from Weaviate
        """
        query_embedding = self.model.encode(query).tolist()
        
        # Retrieve more results initially for filtering
        initial_limit = 100  # Get more results initially for better filtering
        
        response = self.collection.query.near_vector(
            near_vector=query_embedding,
            limit=initial_limit,
            return_metadata=["distance"],
            return_properties=[self.embedding_field],
        )
        
        # Filter by distance and informative content
        filtered_results = [
            doc for doc in response.objects 
            if doc.metadata.distance <= distance_threshold 
            and self.is_informative_content(doc.properties[self.embedding_field], query)
        ]
        
        # If we have too few results after filtering, get the original results
        # if len(filtered_results) < 2:
        #     filtered_results = [
        #         doc for doc in response.objects 
        #         if doc.metadata.distance <= distance_threshold
        #     ][:limit]
            
        # Sort by distance and take top results
        filtered_results = sorted(filtered_results, key=lambda x: x.metadata.distance)[:limit]
        return filtered_results

    def generate_prompt(self, query: str, context: List[Dict[Any, Any]]) -> str:
        """
        Create a prompt combining the retrieved context and user query
        """
        # Sort contexts by potential relevance (using distance as a proxy)
        sorted_contexts = sorted(context, key=lambda x: x.metadata.distance)
        context_str = "\n\n".join([doc.properties[self.embedding_field] for doc in sorted_contexts])
        
        prompt = f"""Please answer the question based on the following context:

Context:
{context_str}

Question: {query}

Please provide a detailed answer using only the information from the given context. The provided context is the Whatsapp chats between Shashank Hegde and Shashin Bhaskar. If the context doesn't contain enough information to answer the question, please state that explicitly."""
        
        return prompt

    async def get_completion(self, prompt: str) -> str:
        """
        Get completion from ChatGPT API
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information, clearly state that."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    async def query(self, user_query: str) -> str:
        """
        Main RAG pipeline
        """
        try:
            # 1. Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(user_query)
            
            if not relevant_docs:
                return "I couldn't find any relevant information to answer your question."
            
            # 2. Generate prompt with context
            prompt = self.generate_prompt(user_query, relevant_docs)
            print(f"Prompt {prompt}")
            
            # 3. Get completion from ChatGPT
            response = await self.get_completion(prompt)
            return response
        finally:
            self.client.close()

# Usage example
async def main():
    rag = RAGSystem(
        openai_api_key="openai-key",
        class_name="WhatsAppMessage"
    )
    
    query = "Example Prompt"
    response = await rag.query(query)
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



