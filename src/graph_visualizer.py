import json

def render_interactive_graph(graph_data: dict, height="520px") -> str:
    """
    Renders an interactive vis-network graph directly from JSON nodes/edges.
    Self-contained, fast, and does not glitch inside Streamlit iframes.
    """
    if not graph_data or not graph_data.get("nodes"):
        return ""

    nodes_list = []
    for node in graph_data.get("nodes", []):
        nodes_list.append({
            "id": node,
            "label": node,
            "color": {"background": "#1b5e20", "border": "#4caf50"},
            "font": {"color": "#ffffff", "size": 15},
            "shape": "box",
            "margin": 10
        })

    edges_list = []
    for edge in graph_data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label", edge.get("relation", ""))
        if source in graph_data.get("nodes", []) and target in graph_data.get("nodes", []):
            edges_list.append({
                "from": source,
                "to": target,
                "label": label,
                "arrows": "to",
                "color": {"color": "#81c784", "highlight": "#a5d6a7"},
                "font": {"color": "#b0bec5", "size": 11, "align": "horizontal"}
            })

    nodes_json = json.dumps(nodes_list)
    edges_json = json.dumps(edges_list)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style type="text/css">
        #network-container {{
          width: 100%;
          height: {height};
          background-color: #121212;
          border-radius: 8px;
          border: 1px solid #333333;
        }}
      </style>
    </head>
    <body>
      <div id="network-container"></div>
      <script type="text/javascript">
        const container = document.getElementById('network-container');
        const data = {{
          nodes: new vis.DataSet({nodes_json}),
          edges: new vis.DataSet({edges_json})
        }};
        const options = {{
          physics: {{
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{
              gravitationalConstant: -50,
              centralGravity: 0.01,
              springLength: 100,
              springConstant: 0.08
            }}
          }},
          interaction: {{
            hover: true,
            zoomView: true,
            dragView: true
          }}
        }};
        new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    return html_content