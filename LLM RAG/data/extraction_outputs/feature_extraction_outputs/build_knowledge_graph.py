import pandas as pd
import networkx as nx
from pyvis.network import Network
import math

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('popular_features_by_place.csv')
        poi_df = pd.read_csv('poi_data.csv')
        poi_info = poi_df.set_index('poi_name').to_dict('index')
    except FileNotFoundError:
        print("Error: Required CSV files not found.")
        return

    # Create a networkx graph
    G = nx.Graph()

    print("Building graph nodes and edges...")
    for _, row in df.iterrows():
        place = row['place_name']
        feature = row['feature']
        weight = row['review_percentage']
        sentiment = row['dominant_sentiment']
        total_revs = row['total_reviews']
        
        # Get POI information
        p_info = poi_info.get(place, {})
        category = p_info.get('category', 'Uncategorized')
        
        # Handle NaN values for formatting
        if pd.isna(category): category = 'Uncategorized'
        
        # Add Category node (Orange)
        if not G.has_node(category):
            G.add_node(category, group='Category', title=f"<b>Category:</b> {category}", size=25, color='#FF9800')

        # Add Place node (Green)
        if not G.has_node(place):
            # Scale node size logarithmically based on total reviews
            size = 20 + (math.log(max(total_revs, 1)) * 5)
            hover_text = (
                f"<b>Place:</b> {place}<br>"
                f"<b>Category:</b> {category}<br>"
                f"<b>Total Reviews:</b> {total_revs}"
            )
            G.add_node(place, group='Place', title=hover_text, size=size, color='#4CAF50')
            
            # Link Place to its Category
            G.add_edge(place, category, value=1, color='#FFB74D', title="BELONGS_TO")
            
        # Add Feature node (Blue)
        if not G.has_node(feature):
            G.add_node(
                feature, 
                group='Feature', 
                title=f"<b>Feature:</b> {feature}", 
                size=15, 
                color='#2196F3' # Blue
            )
            
        # Determine edge color based on sentiment
        edge_color = '#9E9E9E' # Gray (Neutral)
        if sentiment == 'positive':
            edge_color = '#81C784' # Light Green
        elif sentiment == 'negative':
            edge_color = '#E57373' # Light Red
            
        # Add Edge (thickness based on review percentage)
        edge_value = max(1, weight / 10) # scale down percentage for visual thickness
        G.add_edge(
            place, 
            feature, 
            value=edge_value, 
            title=f"<b>Mentions:</b> {weight}%<br><b>Sentiment:</b> {sentiment}", 
            color=edge_color
        )

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("Generating interactive HTML visualization...")
    # Initialize PyVis network with remote CDN to fix loading issues
    # Increased height for better visibility
    net = Network(
        height='1000px', 
        width='100%', 
        bgcolor='#1E1E1E', 
        font_color='white', 
        select_menu=True, 
        filter_menu=True,
        cdn_resources='remote'
    )

    # Inherit nodes and edges from networkx
    net.from_nx(G)

    # Configure advanced physics for a high-quality, spread-out layout.
    # The stabilization process runs silently before the graph is shown, preventing browser freezing.
    physics_options = """
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -150,
          "centralGravity": 0.005,
          "springLength": 350,
          "springConstant": 0.05,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      }
    }
    """
    net.set_options(physics_options)

    # Save the visualization to an HTML file using UTF-8 encoding
    output_file = 'knowledge_graph.html'
    html_content = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Visualization saved to {output_file}! Open this file in your web browser.")

if __name__ == "__main__":
    main()
