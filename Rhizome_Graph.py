import streamlit as st
import networkx as nx
import random
import plotly.graph_objs as go
from llmcore import node_invoke
import logging
import os
# Set up logging        
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use session state for comm_log to persist across reruns
if 'comm_log' not in st.session_state:
    st.session_state['comm_log'] = []

# Remove global comm_log and use session state in NodeData
class NodeData:
    def __init__(self, label, concept_type, weight):
        self.label = label
        self.concept_type = concept_type
        self.weight = weight
        self.response = None  # Placeholder for LLM response
    def __repr__(self):
        return f"{self.label} ({self.concept_type}, w={self.weight}, response={self.response})"

    def invoke(self, prompt, G=None, node_id=None):
        # Assign a unique color to each node for the log
        node_colors_palette = [
            "#e6194b",  "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
            "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
            "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff"
        ]
        color_self = node_colors_palette[node_id % len(node_colors_palette)]
        color_neighbor = node_colors_palette[(node_id+1) % len(node_colors_palette)]
        if node_id==0:
            self.response = node_invoke(self.label, prompt,role=self.concept_type)
            # Log the communication
            st.session_state['comm_log'].append(f"<span style='color:{color_self}'>{self.label}</span> received prompt: {prompt}\n")
            st.session_state['comm_log'].append(f"<span style='color:{color_self}'>{self.label}</span>: {self.response}\n")
            return self.response
        if G is not None and node_id is not None:
            neighbors = list(G.neighbors(node_id))
            st.session_state['comm_log'].append(f"<span style='color: #888'>Neighbors of {self.label}: {sorted(neighbors)}</span>\n")
            for neighbor in sorted(neighbors):
                neighbor_data = G.nodes[neighbor]['data']
                if neighbor_data.response is not None:
                    self.response = node_invoke(self.label, neighbor_data.response,role=self.concept_type)
                    st.session_state['comm_log'].append(
                        f"<span style='color:{color_neighbor}'><b>{neighbor_data.label}</b></span> "
                        f"<span style='color: #888'>→</span> "
                        f"<span style='color:{color_self}'><b>{self.label}</b></span><br>"
                    )
                    st.session_state['comm_log'].append(f"<span style='color:{color_neighbor}'>{neighbor_data.label}</span>'s message to <span style='color:{color_self}'>{self.label}</span>: {neighbor_data.response}\n")
                    st.session_state['comm_log'].append(f"<span style='color:{color_self}'>{self.label}</span>: {self.response}\n")
            return self.response

st.set_page_config(layout="wide")
st.title("Rhizome")
# setting up Google API key in sidebar
with st.sidebar.expander("🔑 Google API Key Setup", expanded=False):
    api_key = st.text_area("Enter your Google API Key", value="")
    if st.button("Set Google API Key"):
        st.session_state["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key  # Set for current process/session only
        st.success("Google API Key set for this session.")
# Sidebar controls
num_nodes = st.sidebar.slider('Number of Nodes', min_value=10, max_value=500, value=3, step=1)
avg_degree = st.sidebar.slider('Average Degree', min_value=1, max_value=50, value=4, step=1)

# Only create the graph if it doesn't already exist in session state
if 'G' not in st.session_state or st.session_state.get('num_nodes') != num_nodes or st.session_state.get('avg_degree') != avg_degree:
    # Create the graph
    G = nx.Graph()
    for i in range(num_nodes):
        label = f"Node-{i}"
        concept_type = random.choice(['Philosopher', 'Analyst', 'Mathematician', 'Tyrant'])
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

    st.session_state['G'] = G
    st.session_state['pos_3d'] = pos_3d
    st.session_state['num_nodes'] = num_nodes
    st.session_state['avg_degree'] = avg_degree
else:
    G = st.session_state['G']
    pos_3d = st.session_state['pos_3d']

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
    'Philosopher': 'red',
    'Analyst': 'green',
    'Mathematician': 'blue',
    'Tyrant': 'orange'
}

x_nodes = [pos_3d[node][0] for node in G.nodes()]
y_nodes = [pos_3d[node][1] for node in G.nodes()]
z_nodes = [pos_3d[node][2] for node in G.nodes()]
labels = [G.nodes[node]['data'].label for node in G.nodes()]
node_colors = [concept_colors[G.nodes[node]['data'].concept_type] for node in G.nodes()]
node_sizes = [8 + 20 * G.nodes[node]['data'].weight for node in G.nodes()]
hover_texts = [
    f"{G.nodes[node]['data'].label}<br>Type: {G.nodes[node]['data'].concept_type}<br>Weight: {G.nodes[node]['data'].weight}<br>Response: {G.nodes[node]['data'].response if G.nodes[node]['data'].response else 'No response yet'}"
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

# Initial plot (before prompt)
st.plotly_chart(fig, use_container_width=True, key='initial_plot')

# --- Prompt Handling ---
st.sidebar.markdown('---')
st.sidebar.subheader('Send Prompt to a Node 0')
user_prompt = st.sidebar.text_input('Enter your prompt for node 0:')

# If a prompt is provided, invoke the node and update the graph
if user_prompt:
    target_node = 0
    node_data = G.nodes[target_node]['data']
    response = node_data.invoke(user_prompt, G, target_node)
    st.sidebar.markdown(f"**Prompt sent to:** {node_data.label}")
    st.sidebar.markdown(f"**LLM Response:** {response}")
    
    logging.info("All nodes in G: %s", list(G.nodes()))
    # Chain reaction: invoke all other nodes with the same prompt
    for n in G.nodes():
        logging.info("Parsing node %s", n)
        if n != target_node:
            logging.info("Parsing node %s", n)
            G.nodes[n]['data'].invoke(user_prompt, G, n)
    
    # Update hover_texts and plot after prompt
    hover_texts = [
        f"{G.nodes[node]['data'].label}<br>Type: {G.nodes[node]['data'].concept_type}<br>Weight: {G.nodes[node]['data'].weight}<br>Response: {G.nodes[node]['data'].response if G.nodes[node]['data'].response else 'No response yet'}"
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
    st.plotly_chart(fig, use_container_width=True, key='after_prompt_plot')

# --- Communication Log Viewer ---
st.sidebar.markdown('---')
st.sidebar.subheader('Communication Log (All Nodes)')

if st.session_state['comm_log']:
    st.sidebar.markdown(
        """
        <style>
        .console-log-area {
            background-color: #111;
            color: #0f0;
            font-family: 'Fira Mono', 'Consolas', 'Menlo', 'Monaco', monospace;
            font-size: 14px;
            padding: 10px;
            border-radius: 6px;
            min-height: 500px;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f"<div class='console-log-area'>{'<br>'.join(st.session_state['comm_log'])}</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.info('No node responses yet. Interact with the network to generate communication.')
