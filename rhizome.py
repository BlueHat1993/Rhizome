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

# Parameters
num_nodes = 1000
avg_degree = 10  # Reduced for less clutter

# Create the graph
G = nx.Graph()

# Add nodes with class-based data
for i in range(num_nodes):
    label = f"Node-{i}"
    concept_type = random.choice(['affect', 'concept', 'assemblage', 'event'])
    weight = round(random.uniform(0.1, 1.0), 2)
    data = NodeData(label, concept_type, weight)
    G.add_node(i, data=data)

# Randomly connect nodes rhizomatically
for node in G.nodes():
    # With 70% probability, connect this node to others (to allow loose ends)
    if random.random() < 0.7:
        possible_targets = [n for n in G.nodes() if n != node and not G.has_edge(node, n)]
        num_edges = random.randint(1, avg_degree)
        # Only add edges if there are enough possible targets
        if possible_targets:
            targets = random.sample(possible_targets, k=min(num_edges, len(possible_targets)))
            for target in targets:
                G.add_edge(node, target)

# Add some extra edges to increase complexity (reduced)
for _ in range(num_nodes // 2):
    n1, n2 = random.sample(list(G.nodes()), 2)
    if not G.has_edge(n1, n2):
        G.add_edge(n1, n2)

# Visualization (Interactive 3D with Plotly)
pos_3d = nx.spring_layout(G, dim=3, iterations=100, seed=42)

# Extract node positions
x_nodes = [pos_3d[node][0] for node in G.nodes()]
y_nodes = [pos_3d[node][1] for node in G.nodes()]
z_nodes = [pos_3d[node][2] for node in G.nodes()]
labels = [G.nodes[node]['data'].label for node in G.nodes()]

# Sample a subset of edges for visualization if too many
all_edges = list(G.edges())
max_edges_to_show = 1000
if len(all_edges) > max_edges_to_show:
    import random
    edges_to_draw = random.sample(all_edges, max_edges_to_show)
else:
    edges_to_draw = all_edges

# Edge coordinates (curved)
curved_edge_x = []
curved_edge_y = []
curved_edge_z = []
num_curve_points = 20  # Number of points per curve

for edge in edges_to_draw:
    x0, y0, z0 = pos_3d[edge[0]]
    x1, y1, z1 = pos_3d[edge[1]]
    # Compute a control point offset from the midpoint for curvature
    mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    # Offset perpendicular to the line for visible curvature
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    # Find a perpendicular vector (arbitrary, but consistent)
    perp = (-dy, dx, 0)
    norm = (perp[0]**2 + perp[1]**2 + perp[2]**2) ** 0.5 or 1
    perp = (perp[0]/norm, perp[1]/norm, perp[2]/norm)
    curve_scale = 0.08  # Adjust for more/less curve
    cx = mx + perp[0] * curve_scale
    cy = my + perp[1] * curve_scale
    cz = mz + perp[2] * curve_scale

    # Generate points along the quadratic Bezier curve
    for t in [i / num_curve_points for i in range(num_curve_points + 1)]:
        xt = (1 - t)**2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
        yt = (1 - t)**2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
        zt = (1 - t)**2 * z0 + 2 * (1 - t) * t * cz + t**2 * z1
        curved_edge_x.append(xt)
        curved_edge_y.append(yt)
        curved_edge_z.append(zt)
    # Add None to break the line between edges
    curved_edge_x.append(None)
    curved_edge_y.append(None)
    curved_edge_z.append(None)

edge_trace = go.Scatter3d(
    x=curved_edge_x, y=curved_edge_y, z=curved_edge_z,
    mode='lines',
    line=dict(color='gray', width=2),
    hoverinfo='none'
)

# Map concept_type to colors
concept_colors = {
    'affect': 'red',
    'concept': 'green',
    'assemblage': 'blue',
    'event': 'orange'
}

node_colors = [concept_colors[G.nodes[node]['data'].concept_type] for node in G.nodes()]
node_sizes = [8 + 20 * G.nodes[node]['data'].weight for node in G.nodes()]

# Add more info to hover text
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
fig.show()
