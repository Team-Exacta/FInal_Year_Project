import sys
import os
import time

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from src.core.config import get_llm
from src.database.chroma_client import ChromaClient
from src.core.logger import get_logger

logger = get_logger("evaluate")

# Try to import the compiled graph
try:
    from src.agents.graph import app
except ImportError:
    app = None

# Initialize standard RAG components
llm = get_llm()
chroma = ChromaClient()
retriever = chroma.get_retriever(k=3)

def run_standard_rag(question: str):
    start_time = time.time()
    
    context = ""
    if retriever:
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        
    prompt = f"Answer the question using the context. Question: {question}\nContext: {context}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    latency = time.time() - start_time
    usage = response.response_metadata.get('token_usage', {})
    
    return response.content, latency, usage

def run_multi_agent_rag(question: str):
    start_time = time.time()
    
    if not app:
        return "App not loaded", 0
        
    inputs = {"question": question, "iterations": 0}
    current_state = inputs.copy()
    
    for output in app.stream(inputs):
        for node_name, state_update in output.items():
            current_state.update(state_update)
            
    latency = time.time() - start_time
    return current_state.get("draft", "No draft found."), latency

def main():
    test_queries = [
        "Which beaches in Colombo are peaceful?",
        "Does Mount Lavinia beach have free parking?",
        "What features does Sigiriya Rock have?"
    ]
    
    logger.info("=== STARTING EVALUATION SCRIPT ===")
    
    for i, query in enumerate(test_queries):
        logger.info(f"--- Query {i+1}: {query} ---")
        
        logger.info("Running Standard RAG...")
        std_ans, std_lat, std_usage = run_standard_rag(query)
        logger.info(f"Standard RAG Latency: {std_lat:.2f} seconds")
        print(f"Standard RAG Answer: {std_ans}\n")
        
        logger.info("Running Multi-Agent RAG...")
        multi_ans, multi_lat = run_multi_agent_rag(query)
        logger.info(f"Multi-Agent RAG Latency: {multi_lat:.2f} seconds")
        print(f"Multi-Agent RAG Answer: {multi_ans}\n")

if __name__ == "__main__":
    main()
