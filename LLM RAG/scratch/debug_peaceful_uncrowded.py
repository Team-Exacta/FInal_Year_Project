import sys
import os

workspace_root = r"c:\LLM RAG\update promts"
sys.path.append(workspace_root)

from src.database.neo4j_client import Neo4jClient

def main():
    db = Neo4jClient()
    print("Connected to Neo4j.")
    
    cypher = """
    MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature)
    WHERE f.name = 'Peacefulness' AND r.sentiment = 'positive'
    AND NOT EXISTS {
        MATCH (p)-[r2:HAS_FEATURE]->(f2:Feature)
        WHERE f2.name = 'Crowdedness' AND r2.sentiment = 'negative' AND r2.percentage > 20.0
    }
    OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
    RETURN p.name AS place, d.name AS district, r.percentage AS peacefulness_pct
    ORDER BY r.percentage DESC
    """
    results = db.query(cypher)
    print("\nPeaceful places with NO high crowdedness (>20%):")
    for r in results:
        print(f"Place: {r['place']} | District: {r['district']} | Peacefulness %: {r['peacefulness_pct']}")

if __name__ == "__main__":
    main()
