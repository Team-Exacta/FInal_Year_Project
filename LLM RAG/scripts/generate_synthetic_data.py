import sys
import os
import json
from langchain_core.messages import HumanMessage

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import get_llm
from src.database.neo4j_client import Neo4jClient
from src.core.logger import get_logger

logger = get_logger("generate_synthetic_data")

def main():
    logger.info("Starting synthetic data generation")
    
    try:
        neo4j = Neo4jClient()
        llm = get_llm()
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        print(f"Error: Failed to initialize clients: {e}")
        return
        
    # Query for some facts
    cypher = """
    MATCH (p:Place)-[r:HAS_ACTIVITY]->(a:Activity)
    WHERE r.percentage > 50
    RETURN p.name AS place, a.name AS activity, r.percentage AS percentage
    LIMIT 10
    """
    
    try:
        results = neo4j.query(cypher)
        logger.info(f"Retrieved {len(results)} facts from Neo4j")
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}")
        results = []
        
    if not results:
        logger.warning("No facts found in Neo4j or query failed. Using dummy data for testing.")
        print("Warning: No facts found in Neo4j. Using fallback data.")
        results = [
            {"place": "Ella", "activity": "Hiking", "percentage": 85.0},
            {"place": "Arugam Bay", "activity": "Surfing", "percentage": 90.0},
            {"place": "Mirissa", "activity": "Whale Watching", "percentage": 75.0}
        ]
        
    synthetic_data = []
    
    for i, res in enumerate(results):
        place = res.get("place")
        activity = res.get("activity")
        percentage = res.get("percentage")
        
        prompt = f"""
        You are an expert at generating evaluation datasets for RAG systems.
        Based on the following fact about Sri Lanka tourism:
        - Place: {place}
        - Activity: {activity}
        - Evidence Percentage: {percentage}% of reviewers mentioned this positively.
        
        Generate a realistic user question that a tourist might ask, where the answer would be based on this fact.
        Also provide the ground truth answer derived from the fact.
        
        Return the result in JSON format EXACTLY like this:
        {{
            "question": "Where is a good place for hiking in Sri Lanka?",
            "ground_truth": "Ella is a highly recommended place for hiking, with 85% of reviews supporting it."
        }}
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                ).strip()
            else:
                content = str(content).strip()
                
            # Simple cleanup of markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                
            parsed = json.loads(content)
            parsed["source_fact"] = res
            synthetic_data.append(parsed)
            logger.info(f"Generated question {i+1}/{len(results)}")
            print(f"Generated question {i+1}/{len(results)}")
            
        except Exception as e:
            logger.error(f"Failed to generate for {place}: {e}")
            print(f"Failed to generate for {place}: {e}")
            
    # Save to file
    output_dir = "data/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "synthetic_test_questions.json")
    
    try:
        with open(output_file, "w") as f:
            json.dump(synthetic_data, f, indent=4)
        logger.info(f"Saved {len(synthetic_data)} questions to {output_file}")
        print(f"\nSuccess! Saved {len(synthetic_data)} questions to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
