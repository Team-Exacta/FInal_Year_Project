from src.database.neo4j_client import Neo4jClient

def main():
    try:
        client = Neo4jClient()
        res = client.query("MATCH (a:Activity) RETURN count(a) as count")
        print(f"Number of Activity nodes in database: {res}")
        
        # Also check for relationships
        res_rel = client.query("MATCH ()-[r:HAS_ACTIVITY]->() RETURN count(r) as count")
        print(f"Number of HAS_ACTIVITY relationships: {res_rel}")
        
        client.close()
    except Exception as e:
        print(f"Error connecting to Neo4j or running query: {e}")

if __name__ == "__main__":
    main()
