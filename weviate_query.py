import weaviate
import re
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer
from weaviate.collections.classes.filters import Filter


class WhatsAppChatQuerier:
    def __init__(self):
        self.client = weaviate.connect_to_local()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.collections.get("WhatsAppMessage")

    def search_by_text(self, query_text: str, 
        limit: int = 5,
        certainty: float = 0.5,  # Lowered threshold for testing
        include_vector: bool = False):
        """Semantic search using text"""
        query_vector = self.model.encode(query_text).tolist()
        
        # Add some debugging prints
        print(f"Query vector dimension: {len(query_vector)}")
        
        # Try without certainty first
        results = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            include_vector=include_vector
        )
        
        if not results.objects:
            print("No results found with vector search")
            # Get a sample of vectors from the database for comparison
            sample = self.collection.query.fetch_objects(
                limit=1,
                include_vector=True
            )
            if sample.objects:
                print(f"Sample vector dimension: {len(sample.objects[0].vector)}")
                print(f"Embedding vector dimension: {len(sample.objects[0].properties['properties']['embedding'])}")

        
        return results
    
    def get_schema_info(self):
        """Print schema information to debug"""
        collection_info = self.collection.config.get()
        print("Collection Properties:")
        for prop in collection_info.properties:
            print(f"- {prop.name}: {prop.data_type}")

    def search_by_time_range(self, start_date: datetime, end_date: datetime, limit: int = 5):
        """Search messages within a time range"""
        time_filter = Filter.by_property("timestamp").greater_or_equal(start_date.isoformat())\
            .__and__(Filter.by_property("timestamp").less_or_equal(end_date.isoformat()))

        result = self.collection.query.fetch_objects(
            limit=limit,
            filters=time_filter
        )
        return result

    def search_by_sender(self, sender_name, limit=5):
        """Search messages from a specific sender"""
        result = self.collection.query.fetch_objects(
            limit=limit,
            filters={
                "path": ["sender"],
                "operator": "Equal",
                "valueString": sender_name
            }
        )
        return result

# Usage example
if __name__ == "__main__":
    from datetime import datetime, timedelta
    querier = WhatsAppChatQuerier()

    print("\nChecking schema:")
    querier.get_schema_info()
    # print(querier.collection.query.fetch_objects(
    #     limit=5,
    #     include_vector=True
    # ))
    # Example 1: Semantic search
    print("\nSemantic search for 'meeting tomorrow':")
    results = querier.search_by_text("bro comment the line where you are calling get_respse_from_gpt")
    for msg in results.objects:
        print(f"Sender: {msg.properties['sender']}")
        print(f"Message: {msg.properties['content']}")
        print(f"Time: {msg.properties['timestamp']}")
        print("---")

    # Example 2: Time range search (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    print("\nMessages from last 7 days:")
    # results = querier.search_by_time_range(start_date, end_date)
    # for msg in results:
    #     print(f"Sender: {msg.properties['sender']}")
    #     print(f"Message: {msg.properties['content']}")
    #     print(f"Time: {msg.properties['timestamp']}")
    #     print("---")

    # # Example 3: Search by sender
    print("\nMessages from specific sender:")
    results = querier.search_by_sender("John")  # Replace with actual sender name
    for msg in results:
        print(f"Sender: {msg.properties['sender']}")
        print(f"Message: {msg.properties['content']}")
        print(f"Time: {msg.properties['timestamp']}")
        print("---")