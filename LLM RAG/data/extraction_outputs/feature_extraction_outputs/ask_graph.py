import pandas as pd
import os
import google.generativeai as genai
import re

def configure_ai():
    print("=== Knowledge Graph AI Query Engine ===")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please enter your Google Gemini API Key: ").strip()
    
    try:
        genai.configure(api_key=api_key)
        # Test the connection
        model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
        return model
    except Exception as e:
        print(f"Failed to configure API: {e}")
        return None

def extract_python_code(response_text):
    # Regex to find code block
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Fallback: if no markdown formatting, assume the whole string is code (if it doesn't look like plain text)
    if "=" in response_text and "final_answer" in response_text:
        return response_text.replace("```", "")
    return None

def main():
    # Load Data
    try:
        features_df = pd.read_csv('popular_features_by_place.csv')
        poi_df = pd.read_csv('poi_data.csv')
    except FileNotFoundError:
        print("Error: Required CSV files not found. Please run this in the correct directory.")
        return

    model = configure_ai()
    if not model:
        return

    print("\nData loaded successfully! 261 Places and 1416 Features are ready.")
    print("Type 'exit' or 'quit' to stop.\n")

    unique_features = list(features_df['feature'].dropna().unique())
    unique_categories = list(poi_df['category'].dropna().unique())

    system_instructions = f"""You are a Data Analyst AI working with a Travel Knowledge Graph.
You have access to two pandas DataFrames:
1. `features_df`: Columns are {list(features_df.columns)}. It maps places to their features, review percentage, and sentiment ('positive', 'negative', 'neutral').
2. `poi_df`: Columns are {list(poi_df.columns)}. It maps places to their category, city, district, province, and coordinates.

AVAILABLE CATEGORIES (Sample): {unique_categories[:20]}
AVAILABLE FEATURES (Sample): {unique_features[:30]}

The user will ask a question in plain English. Your job is to write a Python snippet that calculates the exact answer.
Rules for your code:
1. Assume `features_df` and `poi_df` are already loaded variables. Do NOT import pandas or read csv files.
2. Calculate the answer, format it into a readable string, and assign that string to a variable named `final_answer`.
3. Do NOT use print(). Only assign to `final_answer`.
4. Output ONLY the raw Python code inside a ```python block. Do not add any conversational text.
5. CRITICAL: ALWAYS use case-insensitive substring matching (e.g., `str.contains('keyword', case=False, na=False)`) when filtering features or categories.
6. SEMANTIC MAPPING: If the user asks for a concept, map it to the closest matching items from the AVAILABLE FEATURES or CATEGORIES. Do NOT search for strings that do not exist in the available lists.
7. GEOGRAPHY: If a user asks for a place in a specific location (like "Colombo"), always use case-insensitive substring matching across `city`, `district`, AND `province` columns, as the location might be stored as "Colombo District" rather than the exact city name.

Example output:
```python
# Code to answer the question
mask = features_df['feature'].str.contains('waterfall', case=False, na=False)
result = features_df[mask]['place_name'].tolist()
final_answer = "The waterfalls are: " + ", ".join(result)
```
"""

    while True:
        query = input("\nAsk your Knowledge Graph: ")
        if query.lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue

        print("Thinking...")
        
        prompt = system_instructions + "\n\nUser Question: " + query
        
        try:
            response = model.generate_content(prompt)
            code = extract_python_code(response.text)
            
            if not code:
                print("Error: The AI didn't return valid Python code.")
                continue
            
            # Execute the code in a local environment
            local_env = {'features_df': features_df, 'poi_df': poi_df, 'pd': pd}
            
            try:
                exec(code, {}, local_env)
                answer = local_env.get('final_answer')
                if answer:
                    print("\n--- Answer ---")
                    print(answer)
                    print("--------------")
                else:
                    print("\n[The AI ran the search, but didn't output a final_answer variable.]")
            except Exception as code_error:
                print(f"\n[Error executing the AI's search code: {code_error}]")
                # print("Code generated was:\n", code)
                
        except Exception as e:
            print(f"\n[API Error: {e}]")

if __name__ == "__main__":
    main()
