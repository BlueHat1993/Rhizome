import streamlit as st
import networkx as nx
import random
import plotly.graph_objs as go

# Define a class for node data
class NodeData:
    def __init__(self, label, concept_type, weight):
        self.label = label
        self.concept_type = concept_type
        self.weight = weight
    def __repr__(self):
        return f"{self.label} ({self.concept_type}, w={self.weight})"

st.set_page_config(layout="wide")
st.title("Rhizomatic Network (Interactive 3D)")

# Sidebar controls
num_nodes = st.sidebar.slider('Number of Nodes', min_value=10, max_value=500, value=100, step=1)
avg_degree = st.sidebar.slider('Average Degree', min_value=1, max_value=50, value=4, step=1)

# Create the graph
G = nx.Graph()
for i in range(num_nodes):
    label = f"Node-{i}"
    concept_type = random.choice(['affect', 'concept', 'assemblage', 'event'])
    weight = round(random.uniform(0.1, 1.0), 2)
    data = NodeData(label, concept_type, weight)
    G.add_node(i, data=data)

for node in G.nodes():
    if random.random() < 0.7:
        possible_targets = [n for n in G.nodes() if n != node and not G.has_edge(node, n)]
        num_edges = random.randint(1, avg_degree)
        if possible_targets:
            targets = random.sample(possible_targets, k=min(num_edges, len(possible_targets)))
            for target in targets:
                G.add_edge(node, target)
for _ in range(num_nodes // 2):
    n1, n2 = random.sample(list(G.nodes()), 2)
    if not G.has_edge(n1, n2):
        G.add_edge(n1, n2)

pos_3d = nx.spring_layout(G, dim=3, iterations=100, seed=42)

# Sample a subset of edges for visualization if too many
all_edges = list(G.edges())
max_edges_to_show = 1000
if len(all_edges) > max_edges_to_show:
    edges_to_draw = random.sample(all_edges, max_edges_to_show)
else:
    edges_to_draw = all_edges

# Edge coordinates (straight lines)
edge_x, edge_y, edge_z = [], [], []
for edge in edges_to_draw:
    x0, y0, z0 = pos_3d[edge[0]]
    x1, y1, z1 = pos_3d[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='gray', width=2),
    hoverinfo='none'
)

concept_colors = {
    'affect': 'red',
    'concept': 'green',
    'assemblage': 'blue',
    'event': 'orange'
}

x_nodes = [pos_3d[node][0] for node in G.nodes()]
y_nodes = [pos_3d[node][1] for node in G.nodes()]
z_nodes = [pos_3d[node][2] for node in G.nodes()]
labels = [G.nodes[node]['data'].label for node in G.nodes()]
node_colors = [concept_colors[G.nodes[node]['data'].concept_type] for node in G.nodes()]
node_sizes = [8 + 20 * G.nodes[node]['data'].weight for node in G.nodes()]
hover_texts = [
    f"{G.nodes[node]['data'].label}<br>Type: {G.nodes[node]['data'].concept_type}<br>Weight: {G.nodes[node]['data'].weight}"
    for node in G.nodes()
]

node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(
        size=node_sizes,
        color=node_colors,
        line=dict(width=1, color='darkblue')
    ),
    text=labels,
    textposition='top center',
    hoverinfo='text',
    hovertext=hover_texts
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title='Rhizomatic Network (Interactive 3D)',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            bgcolor='black'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=40)
    )
)

st.plotly_chart(fig, use_container_width=True)
