import http.server
import socketserver
import json
import os
import sys

# Ensure root directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.run_rag_v2 import process_query
except ImportError:
    print("Error: Could not import process_query from scripts.run_rag_v2.")
    print("Please ensure you are running this from the root directory or the scripts folder.")
    sys.exit(1)

PORT = 8000
UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui'))

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            question = data.get('question', '')
            print(f"API Received Question: {question}")
            
            # Call the RAG process
            try:
                result = process_query(question)
                
                # process_query now returns {"text": ..., "structured_facts": [...]}
                if isinstance(result, dict):
                    response_text = result.get("text", "")
                    structured_facts = result.get("structured_facts", [])
                else:
                    response_text = result
                    structured_facts = []

                # Extract plain text if LangChain returned a list of message blocks
                if isinstance(response_text, list):
                    texts = []
                    for item in response_text:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                    if texts:
                        response_text = " ".join(texts)
                
                if not isinstance(response_text, str):
                    response_text = str(response_text)

                # Ensure facts are JSON-serializable
                try:
                    json.dumps(structured_facts)
                except Exception:
                    structured_facts = []
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                response_text = "Error processing query on the server."
                structured_facts = []
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response_data = json.dumps({'response': response_text, 'evidence': structured_facts})
            self.wfile.write(response_data.encode('utf-8'))
            
        else:
            self.send_error(404, "Not Found")

def run():
    # Allow address reuse to avoid "Address already in use" errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Sri Lanka Tourism RAG v2 — UI Server")
        print(f"Server started at http://localhost:{PORT}")
        print(f"Serving files from: {UI_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close()

if __name__ == "__main__":
    run()
