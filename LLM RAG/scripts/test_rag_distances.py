import sys
import os

# Adjust sys.path to run from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_rag_v2 import process_query

def test_rag_with_distances():
    print("=" * 60)
    print("TESTING RAG END-TO-END WITH ROAD DISTANCE MATRIX FACTS")
    print("=" * 60)
    
    question = "Which is closer to Mirissa Beach: Weligama Beach or Hiriketiya Beach, and how far are they?"
    
    print(f"Running Query: '{question}'\n")
    result = process_query(question)
    
    print("\n" + "=" * 60)
    print("DETECTED INTENT:")
    intent = result.get("query_intent", {})
    print(f"- Type: {intent.get('intent_type', 'unknown')}")
    print(f"- Confidence: {intent.get('confidence', 0.0):.2f}")
    print(f"- Entities: {intent.get('entities', {})}")
    
    print("\n" + "=" * 60)
    print("RAG FINAL ANSWER:")
    print(result["text"])
    print("=" * 60)

if __name__ == "__main__":
    test_rag_with_distances()
