from src.database.neo4j_client import Neo4jClient

c = Neo4jClient()
c.connect()
print("Maha Saman Dewalaya:")
print(c.query("MATCH (p:Place {name: 'Maha Saman Dewalaya'})-[:LOCATED_IN]->(d:District) RETURN d.name"))
print("Kirinda Viharaya:")
print(c.query("MATCH (p:Place {name: 'Kirinda Viharaya'})-[:LOCATED_IN]->(d:District) RETURN d.name"))
c.close()
