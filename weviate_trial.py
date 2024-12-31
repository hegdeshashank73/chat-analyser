import weaviate
import re
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer

class WhatsAppChatIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize the local sentence transformer
        self.model = SentenceTransformer(model_name)
        self.client = weaviate.connect_to_local()
        self.setup_schema()
    
    def setup_schema(self):
        """Create the schema for WhatsApp messages if it doesn't exist"""
        try:
            collection = self.client.collections.get("WhatsAppMessage")
            print("Collection already exists")
        except Exception as e:
            print("Creating new collection")
            properties = [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The message content",
                    "indexFilterable": True,
                    "indexSearchable": True
                },
                {
                    "name": "sender",
                    "dataType": ["string"],
                    "description": "The sender of the message",
                    "indexFilterable": True,
                    "indexSearchable": True
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the message was sent",
                    "indexFilterable": True
                }
            ]

            self.client.collections.create(
                name="WhatsAppMessage",
                description="Collection storing WhatsApp chat messages",
                properties=properties,
                vectorizer_config={"vectorizer": "none"},  # Changed from None
                vector_index_config={"distance": "cosine"},   # We'll handle vectorization ourselves
            )

    def parse_message(self, line):
        """Parse a single WhatsApp message line"""
        pattern = r'(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)'
        match = re.match(pattern, line)
        
        if match:
            date, time, sender, content = match.groups()
            
            try:
                timestamp = datetime.strptime(f"{date} {time}", "%m/%d/%y %H:%M")
                # Generate embedding for the content
                embedding = self.model.encode(content, convert_to_numpy=True).tolist()
                
                return {
                    "timestamp": timestamp.isoformat(),
                    "sender": sender.strip(),
                    "content": content.strip(),
                    "embedding": embedding
                }
            except Exception as e:
                print(f"Exception received {e}")
                print(f"Received value error for line: {line}")
                return None
        return None

    def index_chat(self, filepath):
        """Index the entire chat file into Weaviate"""
        with open(filepath, 'r', encoding='utf-8') as file:
            collection = self.client.collections.get("WhatsAppMessage")
            
            # Create objects list
            objects_to_insert = []
            total_processed = 0
            
            for line in file:
                line = line.strip()
                if not line or "Messages and calls are end-to-end encrypted" in line:
                    continue
                
                parsed = self.parse_message(line)
                if parsed:
                    objects_to_insert.append(
                        {
                        "properties": {
                            "content": parsed["content"],
                            "sender": parsed["sender"],
                            "timestamp": parsed["timestamp"],
                        },
                        "vector": parsed["embedding"],
                        "class": "WhatsAppMessage"
                    })
                    total_processed += 1
                    
                    # Print progress
                    if total_processed % 100 == 0:
                        print(f"Processed {total_processed} messages")
                
                # Insert in batches of 100
                if len(objects_to_insert) >= 1:
                    try:
                        collection.data.insert(properties={
                            "content": parsed["content"],
                            "sender": parsed["sender"],
                            "timestamp": parsed["timestamp"],
                        }, vector=parsed["embedding"],)
                        # collection.data.insert_many(objects_to_insert)
                        print(f"Inserted batch of {len(objects_to_insert)} messages. Total processed: {total_processed}")
                    except Exception as e:
                        print(f"Error inserting batch: {e}")
                    objects_to_insert = []
            
            # Insert any remaining objects
            if objects_to_insert:
                try:
                    collection.data.insert_many(objects_to_insert)
                    print(f"Inserted final batch of {len(objects_to_insert)} messages. Total processed: {total_processed}")
                except Exception as e:
                    print(f"Error inserting final batch: {e}")


# Usage example
if __name__ == "__main__":
    # Initialize the indexer
    indexer = WhatsAppChatIndexer()
    
    # Index the chat file
    # indexer.index_chat("whatsapp_chat.txt")
    
    # Example query to verify indexing
    try:
        collection = indexer.client.collections.get("WhatsAppMessage")
        result = collection.query.fetch_objects(
            limit=5,
            include_vector=True
        )
        
        print("\nVerification of indexed data:")
        for obj in result.objects:
            print(f"Sender: {obj.properties['sender']}")
            print(f"Message: {obj.properties['content']}")
            print(f"Time: {obj.properties['timestamp']}")
            print(f"Vector: {obj.vector}")
            print("---")
    except Exception as e:
        print(f"Error querying data: {e}")