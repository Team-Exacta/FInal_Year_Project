import sys
sys.path.insert(0, "c:/LLM RAG/update promts")
from scripts.run_rag_v2 import process_query

q = "Where can I go surfing and whale watching in Sri Lanka?"
res = process_query(q)
print("QUESTION:", q)
print("\nANSWER:\n", res.get("text", ""))
