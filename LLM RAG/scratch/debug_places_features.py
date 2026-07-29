import sys
import os

workspace_root = r"c:\LLM RAG\update promts"
sys.path.append(workspace_root)

from src.database.neo4j_client import Neo4jClient

def main():
    db = Neo4jClient()
    print("Connected to Neo4j.")
    
    cypher = """
    MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature {name: 'Crowdedness'})
    RETURN r.sentiment AS sentiment, count(*) AS count
    """
    results = db.query(cypher)
    print("\nCrowdedness sentiments in DB:")
    for r in results:
        print(f"Sentiment: {r['sentiment']} | Count: {r['count']}")

if __name__ == "__main__":
    main()
