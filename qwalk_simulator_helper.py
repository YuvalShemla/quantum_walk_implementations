import numpy as np
from scipy.sparse.linalg import expm
from scipy.sparse import csc_matrix, diags

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, to_rgb

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph





# Function to compute the negative Laplacian matrix
def compute_negative_laplacian(Adj):


    A_copy = Adj.copy()
    A = csc_matrix(A_copy)  # Ensure A is in CSC format for expm efficiency
    degrees = np.array(A.sum(axis=1)).flatten()
    D = diags(degrees, format='csc')  # Create diagonal matrix in CSC format
    L_neg = A - D

    # Here is Akshat's one liner awesome method:
    # for i in range(A.shape[0]):
    #     L[i,i] = -len(A[[i], :].nonzero()[0])

    return L_neg


# Function for Classical Continuous Random Walk
def classical_random_walk(adjacency, initial_position, time_step):
    A = adjacency
    p0 = initial_position
    t = time_step
    L_neg = compute_negative_laplacian(A)

    pos_t = expm(L_neg * t) @ p0
    return pos_t


# Function for Quantum Walk
def quantum_walk(adjacency, initial_position, time_step):
    A_copy = adjacency.copy()
    A = csc_matrix(A_copy)
    psi0 = initial_position
    t = time_step

    psi_t = expm(-1j * A * t) @ psi0
    return psi_t


# Function to compute the difference between quantum and classical probabilities
def get_probability_difference(p_quantum, p_classical):
    return p_quantum - p_classical


def plot_graph(G, labels=False):
    pos = graphviz_layout(G, prog="dot")
    # Draw the graph using positions from G
    nx.draw(
        G,
        pos=pos,
        node_color="#dddddd",
        node_size=250,
        with_labels=labels  # Set to True to display node labels
    )
    plt.show()


def print_state_vectors(prob, label=""):
    n = len(prob)

    print(f"\n==== {label} State Vector ====")
    for i in range(n):
        print(f"Node {i + 1}: {prob[i]:.4f}")



# Function to visualize the graph with node values
def visualize_probabilities(G, prob_values, labels=False, title='', ax=None):
    # Number of nodes
    N = len(prob_values)
    equal_prob = 1 / N

    # Define a custom colormap from gray to pink
    cmap = LinearSegmentedColormap.from_list(
        'CustomMap',
        ['gray', 'purple']
    )

    # Set up normalization with midpoint at equal probability
    norm = TwoSlopeNorm(vmin=0, vcenter=equal_prob, vmax=1)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Map node colors based on probability values
    node_colors = [mapper.to_rgba(p) for p in prob_values]

    # Create a new axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    pos = graphviz_layout(G, prog="dot")

    # Draw the graph with custom colors
    nx.draw(
        G,
        pos=pos,
        node_color=node_colors,
        node_size=250,
        with_labels=labels,
        ax=ax
    )

    # Add color bar with intermediate ticks
    cbar = plt.colorbar(mapper, ax=ax, label="Probability", fraction=0.046, pad=0.04)
    cbar.set_ticks([0, equal_prob, 1])
    cbar.set_ticklabels([f"Low: 0", f"1/N: {equal_prob:.2e}", "High: 1"])

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Remove axis for better layout control
    ax.axis('off')

    # Adjust layout for color bar and graph clarity
    plt.tight_layout()

# ---------------- Less Important Stuff ------------------ #


def create_cayley_tree(d, n):
    """
    Create a Cayley tree with coordination number d and generations n.
    """
    G = nx.Graph()
    node_counter = 0  # Node IDs start from 0

    def add_children(parent, generation):
        nonlocal node_counter
        if generation < n:
            if generation == 0:
                # Root node: d children
                num_children = d
            else:
                # Inner nodes: d-1 children
                num_children = d - 1
            for _ in range(num_children):
                node_counter += 1
                child = node_counter
                G.add_node(child, generation=generation + 1)
                G.add_edge(parent, child)
                add_children(child, generation + 1)
        else:
            # Leaf node; mark it
            G.nodes[parent]['is_leaf'] = True

    # Initialize root node
    root = node_counter
    G.add_node(root, generation=0)
    add_children(root, 0)
    return G


def plot_cayley_tree_central(G):
    # Convert NetworkX graph to PyGraphviz AGraph
    A = to_agraph(G)

    # Customize node attributes
    for node in G.nodes():
        n = A.get_node(node)
        data = G.nodes[node]
        if data.get('is_leaf', False):
            n.attr['color'] = 'green'
            n.attr['style'] = 'filled'
            n.attr['fillcolor'] = 'lightgreen'
        elif data['generation'] == 0:
            # Central node
            n.attr['color'] = 'red'
            n.attr['style'] = 'filled'
            n.attr['fillcolor'] = 'lightcoral'
        else:
            n.attr['color'] = 'blue'
            n.attr['style'] = 'filled'
            n.attr['fillcolor'] = 'lightblue'

    # Customize edge attributes
    for edge in G.edges():
        e = A.get_edge(edge[0], edge[1])
        e.attr['color'] = 'gray'

    # Apply a layout algorithm
    A.layout('twopi')  # Radial layout, central node in the middle

    # Draw the graph to a file and display it inline
    import tempfile
    from IPython.display import Image, display
    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        A.draw(tmpfile.name)
        display(Image(filename=tmpfile.name))
