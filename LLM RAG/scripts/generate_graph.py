"""
scripts/generate_graph.py
Queries the live Neo4j database and generates a fresh, interactive
knowledge_graph.html using vis-network.

Run:
    python -m scripts.generate_graph

Output:
    outputs/knowledge_graph.html  (opens automatically in your browser)
"""

import sys, os, json, webbrowser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_client import Neo4jClient
from src.core.logger import get_logger

logger = get_logger("generate_graph")

NODE_COLORS = {
    "Place":    {"background": "#6C63FF", "border": "#9F99FF", "highlight": {"background": "#8B85FF", "border": "#C0BCFF"}},
    "District": {"background": "#00C2FF", "border": "#66DFFF", "highlight": {"background": "#33CEFF", "border": "#99E9FF"}},
    "Province": {"background": "#FF6B6B", "border": "#FF9999", "highlight": {"background": "#FF8585", "border": "#FFBBBB"}},
    "Category": {"background": "#FFD166", "border": "#FFE0A0", "highlight": {"background": "#FFDA80", "border": "#FFE9BB"}},
    "Feature":  {"background": "#06D6A0", "border": "#80EBD0", "highlight": {"background": "#3DDFB3", "border": "#99EFD8"}},
    "Facility": {"background": "#FF9F43", "border": "#FFC58D", "highlight": {"background": "#FFAF60", "border": "#FFE0C0"}},
}

EDGE_COLORS = {
    "LOCATED_IN":   "#00C2FF",
    "PART_OF":      "#FF6B6B",
    "HAS_CATEGORY": "#FFD166",
    "HAS_FEATURE":  "#06D6A0",
    "HAS_FACILITY": "#FF9F43",
}


def build_graph_data(client: Neo4jClient):
    nodes_map = {}
    edges = []
    counter = [0]

    def add_node(nid, label, node_label, title=""):
        if nid not in nodes_map:
            color = NODE_COLORS.get(node_label, {"background": "#aaa", "border": "#888"})
            nodes_map[nid] = {
                "id": nid, "label": label[:30], "title": title or label,
                "group": node_label, "color": color,
                "font": {"color": "#ffffff", "size": 13},
                "shape": "dot" if node_label in ["Feature", "Facility"] else "ellipse",
                "size": 20 if node_label == "Place" else 14,
            }

    def add_edge(frm, to, label, color_key):
        edges.append({"id": f"e{counter[0]}", "from": frm, "to": to, "label": label,
                      "color": {"color": EDGE_COLORS[color_key]}})
        counter[0] += 1

    logger.info("Querying geographic hierarchy...")
    for r in client.query(
        "MATCH (p:Place)-[:LOCATED_IN]->(d:District)-[:PART_OF]->(pr:Province) "
        "RETURN p.name AS place, p.category AS cat, d.name AS district, pr.name AS province"
    ):
        add_node(r["place"],    r["place"],    "Place",    "Category: " + str(r.get("cat", "")))
        add_node(r["district"], r["district"], "District")
        add_node(r["province"], r["province"], "Province")
        add_edge(r["place"],    r["district"], "IN",      "LOCATED_IN")
        add_edge(r["district"], r["province"], "PART OF", "PART_OF")

    logger.info("Querying categories...")
    for r in client.query(
        "MATCH (p:Place)-[:HAS_CATEGORY]->(c:Category) RETURN p.name AS place, c.name AS cat"
    ):
        add_node(r["place"], r["place"], "Place")
        add_node(r["cat"],   r["cat"],   "Category")
        add_edge(r["place"], r["cat"], "IS", "HAS_CATEGORY")

    logger.info("Querying top-5 features per place...")
    for r in client.query(
        "MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature) "
        "WITH p, f, r ORDER BY r.percentage DESC "
        "WITH p, collect({feat: f.name, pct: r.percentage, sent: r.sentiment})[..5] AS top "
        "UNWIND top AS t "
        "RETURN p.name AS place, t.feat AS feature, t.pct AS pct, t.sent AS sentiment"
    ):
        pct = round(float(r.get("pct") or 0) * 100, 1)
        add_node(r["place"],   r["place"],   "Place")
        add_node(r["feature"], r["feature"], "Feature",
                 title=str(r["feature"]) + " | sentiment: " + str(r.get("sentiment","")) + " | pct: " + str(pct) + "%")
        add_edge(r["place"], r["feature"], str(r.get("sentiment", "")), "HAS_FEATURE")

    logger.info("Querying top-5 facilities per place...")
    for r in client.query(
        "MATCH (p:Place)-[r:HAS_FACILITY]->(fa:Facility) "
        "WITH p, fa, r ORDER BY r.percentage DESC "
        "WITH p, collect({fac: fa.name, pct: r.percentage, sent: r.sentiment})[..5] AS top "
        "UNWIND top AS t "
        "RETURN p.name AS place, t.fac AS facility, t.pct AS pct, t.sent AS sentiment"
    ):
        pct = round(float(r.get("pct") or 0) * 100, 1)
        add_node(r["place"],    r["place"],    "Place")
        add_node(r["facility"], r["facility"], "Facility",
                 title=str(r["facility"]) + " | sentiment: " + str(r.get("sentiment","")) + " | pct: " + str(pct) + "%")
        add_edge(r["place"], r["facility"], str(r.get("sentiment", "")), "HAS_FACILITY")

    logger.info(f"Graph built: {len(nodes_map)} nodes, {len(edges)} edges.")
    return list(nodes_map.values()), edges


def write_html(node_list, edge_list, output_path):
    nodes_j = json.dumps(node_list, ensure_ascii=False)
    edges_j = json.dumps(edge_list, ensure_ascii=False)

    # Build HTML using direct Python string construction — no .format() to avoid brace conflicts
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sri Lanka Knowledge Graph - Live View</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous"/>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
header { background: linear-gradient(135deg, #1a1a2e, #16213e); border-bottom: 1px solid #2a2a5a; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
header h1 { font-size: 1.15rem; color: #a89cff; }
header p  { font-size: 0.76rem; color: #888; margin-top: 2px; }
#toolbar { background: #12122a; border-bottom: 1px solid #222244; padding: 8px 20px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; flex-shrink: 0; }
#toolbar label { font-size: 0.8rem; color: #aaa; }
#toolbar input, #toolbar select { background: #1e1e3a; border: 1px solid #3a3a6a; color: #e0e0e0; border-radius: 6px; padding: 5px 10px; font-size: 0.8rem; outline: none; }
#toolbar button { background: #6C63FF; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 0.8rem; cursor: pointer; }
#toolbar button:hover { background: #5a52e0; }
#stats { font-size: 0.76rem; color: #66ff88; margin-left: auto; }
#legend { display: flex; gap: 16px; align-items: center; background: #12122a; border-bottom: 1px solid #222244; padding: 7px 20px; flex-shrink: 0; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.76rem; color: #ccc; }
.legend-dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
#network { height: 75vh; width: 100%; background: #0f0f1a; }
#info-panel { position: fixed; right: 16px; top: 130px; width: 240px; background: #1a1a2e; border: 1px solid #2a2a5a; border-radius: 10px; padding: 14px; font-size: 0.8rem; display: none; box-shadow: 0 8px 24px rgba(0,0,0,.5); z-index: 999; }
#info-panel h3 { color: #a89cff; margin-bottom: 8px; font-size: 0.88rem; }
#info-panel p { color: #bbb; line-height: 1.6; }
.close-btn { float: right; cursor: pointer; color: #666; font-size: 1rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Sri Lanka - Knowledge Graph (Live)</h1>
    <p>Generated from Neo4j &middot; """)

    parts.append(str(len(node_list)))
    parts.append(""" nodes &middot; """)
    parts.append(str(len(edge_list)))
    parts.append(""" edges</p>
  </div>
</header>

<div id="toolbar">
  <label>Search node: <input id="search-input" type="text" placeholder="e.g. Arugam Bay" style="width:200px;"></label>
  <label>Filter type:
    <select id="type-filter">
      <option value="all">All types</option>
      <option value="Place">Place</option>
      <option value="District">District</option>
      <option value="Province">Province</option>
      <option value="Category">Category</option>
      <option value="Feature">Feature</option>
      <option value="Facility">Facility</option>
    </select>
  </label>
  <button onclick="resetView()">Reset View</button>
  <span id="stats"></span>
</div>

<div id="legend">
  <span style="font-size:0.76rem;color:#888;">Legend:</span>
  <div class="legend-item"><div class="legend-dot" style="background:#6C63FF"></div>Place</div>
  <div class="legend-item"><div class="legend-dot" style="background:#00C2FF"></div>District</div>
  <div class="legend-item"><div class="legend-dot" style="background:#FF6B6B"></div>Province</div>
  <div class="legend-item"><div class="legend-dot" style="background:#FFD166"></div>Category</div>
  <div class="legend-item"><div class="legend-dot" style="background:#06D6A0"></div>Feature</div>
  <div class="legend-item"><div class="legend-dot" style="background:#FF9F43"></div>Facility</div>
</div>

<!-- Removed loading overlay -->

<div id="network"></div>

<div id="info-panel">
  <span class="close-btn" onclick="document.getElementById('info-panel').style.display='none'">x</span>
  <h3 id="info-title">Node details</h3>
  <p id="info-body"></p>
</div>

<script>
var nodesData = """)

    parts.append(nodes_j)
    parts.append(""";
var edgesData = """)
    parts.append(edges_j)
    parts.append(""";

var nodes = new vis.DataSet(nodesData);
var edges = new vis.DataSet(edgesData);

var container = document.getElementById('network');
var options = {
  physics: {
    enabled: true,
    solver: 'barnesHut',
    barnesHut: { gravitationalConstant: -8000, centralGravity: 0.3, springLength: 100, springConstant: 0.04, damping: 0.5, avoidOverlap: 0.5 },
    stabilization: { enabled: false }
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.4 } },
    font: { size: 9, color: '#888', align: 'middle', strokeWidth: 0 },
    smooth: { type: 'continuous' },
    width: 1.0,
    color: { opacity: 0.6 }
  },
  nodes: {
    borderWidth: 2,
    shadow: { enabled: false }
  },
  interaction: {
    hover: true,
    tooltipDelay: 80,
    navigationButtons: true,
    keyboard: true
  }
};

var network = new vis.Network(container, { nodes: nodes, edges: edges }, options);
network.fit();

network.on('click', function(params) {
  if (params.nodes.length > 0) {
    var n = nodes.get(params.nodes[0]);
    document.getElementById('info-title').textContent = n.label;
    document.getElementById('info-body').innerHTML =
      '<b>Type:</b> ' + n.group + '<br><b>Details:</b> ' + (n.title || '-');
    document.getElementById('info-panel').style.display = 'block';
  }
});

network.once('stabilizationIterationsDone', function() {
  document.getElementById('stats').textContent =
    'Rendered ' + nodes.length + ' nodes, ' + edges.length + ' edges';
});

document.getElementById('search-input').addEventListener('input', function() {
  var q = this.value.toLowerCase();
  if (!q) { network.unselectAll(); return; }
  var matched = nodes.get().filter(function(n) { return n.label.toLowerCase().indexOf(q) !== -1; }).map(function(n) { return n.id; });
  network.selectNodes(matched);
  if (matched.length === 1) { network.focus(matched[0], { scale: 1.4, animation: true }); }
});

document.getElementById('type-filter').addEventListener('change', function() {
  var type = this.value;
  nodes.get().forEach(function(n) {
    nodes.update({ id: n.id, hidden: type !== 'all' && n.group !== type });
  });
});

function resetView() {
  document.getElementById('search-input').value = '';
  document.getElementById('type-filter').value = 'all';
  nodes.get().forEach(function(n) { nodes.update({ id: n.id, hidden: false }); });
  network.fit({ animation: true });
  network.unselectAll();
}
</script>
</body>
</html>""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "knowledge_graph.html")

    logger.info("Connecting to Neo4j...")
    client = Neo4jClient()
    client.connect()

    node_list, edge_list = build_graph_data(client)
    client.close()

    write_html(node_list, edge_list, output_path)
    logger.info(f"Graph saved -> {output_path}")
    logger.info("Opening in browser...")
    webbrowser.open(f"file:///{output_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
