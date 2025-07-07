import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from scipy.spatial import ConvexHull
from gudhi import SimplexTree, RipsComplex
import random
import copy
import pandas as pd

np.random.seed(42)
random.seed(42)


def visualize_filtration(G, pos, title):
    """Visualize network evolution across filtration steps (descending weights)"""

    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    filtration_steps = np.arange(0.9, 0.0, -0.1)

    fig, axs = plt.subplots(2, 5, figsize=(22, 6))
    axs = axs.flatten()
    fig.suptitle(title + " Filtration", fontsize=22, y=1.05)

    nodes = list(G.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    st = SimplexTree()
    for node in nodes:
        st.insert([node_index[node]], filtration=0.0)

    for u, v, data in G.edges(data=True):
        filtration_val = 1 - data['weight']
        st.insert([node_index[u], node_index[v]], filtration=filtration_val)

    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 3:
            i, j, k = [node_index[n] for n in clique]
            max_filt = max(
                1 - G[u][v]['weight']
                for u,v in [(clique[0], clique[1]), (clique[1], clique[2]), (clique[0], clique[2])]
            )
            st.insert(sorted([i,j,k]), filtration=max_filt)

    st.initialize_filtration()
    diag = st.persistence(homology_coeff_field=2)

    h1_generators = [(birth, death) for dim, (birth, death) in diag if dim == 1]

    for step_idx, threshold in enumerate(filtration_steps):
        ax = axs[step_idx]

        added_edges = [(u,v) for u,v,d in edges_sorted if d['weight'] >= threshold]
        subG = nx.Graph()
        subG.add_nodes_from(G.nodes)
        subG.add_edges_from(added_edges)

        triangles = [c for c in nx.enumerate_all_cliques(subG) if len(c) == 3]

        nx.draw_networkx_nodes(subG, pos, node_size=120, ax=ax, node_color='lightgray', edgecolors='black')
        nx.draw_networkx_labels(subG, pos, font_size=8, ax=ax)
        nx.draw_networkx_edges(subG, pos, edgelist=added_edges, width=1.5, ax=ax, edge_color='dimgray')

        # Triangles in blue
        for tri in triangles:
            tri_pos = [pos[n] for n in tri]
            poly = Polygon(tri_pos, closed=True, alpha=0.4, edgecolor='blue', facecolor='lightblue', zorder=0)
            ax.add_patch(poly)

        # H1 generators active in light pink
        gudhi_filt = 1 - threshold
        active_generators = 0
        cycles = nx.cycle_basis(subG)

        for birth, death in h1_generators:
            if birth <= gudhi_filt < death:
                active_generators += 1
                for cycle in cycles:
                    if len(cycle) >= 4:
                        try:
                            cycle_pos = [pos[n] for n in cycle]
                            hull = ConvexHull(cycle_pos)
                            poly = Polygon([cycle_pos[i] for i in hull.vertices], closed=True,
                                           alpha=0.3, edgecolor='black', facecolor='lightpink', zorder=0)
                            ax.add_patch(poly)
                        except:
                            poly = Polygon(cycle_pos, closed=True, alpha=0.3,
                                           edgecolor='black', facecolor='lightpink', zorder=0)
                            ax.add_patch(poly)

        ax.set_title(f"Threshold ≥ {threshold:.2f}\nEdges: {len(added_edges)} | Triangles: {len(triangles)} | Weighted Holes: {active_generators}", fontsize=14)
        ax.set_axis_off()
        ax.set_aspect('equal')

    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label='2-Simplices (Triangles)', alpha=0.4),
        Patch(facecolor='lightpink', edgecolor='black', label='Weighted Holes (H1 generators)', alpha=0.3),
    ]

    axs[-1].legend(handles=legend_elements, loc='center', fontsize=12, frameon=False)
    axs[-1].axis('off')
    axs[-1].set_frame_on(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.3)
    plt.show()


def plot_network(G, title, pos=None):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 4))
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightgray",
        edgecolors="black",
        linewidths=1.5,
        width=weights,
        edge_color="gray",
        font_color="black",
    )

    # Edge weight labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8, font_color="black"
    )

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return pos

def get_persistence_diagram(G, title="", print_diagram=True):
    nodes = list(G.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    dist_matrix = np.full((n, n), np.inf)
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        dist_matrix[i, j] = 1 - data["weight"]
        dist_matrix[j, i] = 1 - data["weight"]
    np.fill_diagonal(dist_matrix, 0.0)

    rips_complex = RipsComplex(distance_matrix=dist_matrix, max_edge_length=1.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    simplex_tree.compute_persistence()

    diag = simplex_tree.persistence()
    diag_H0 = [p for p in diag if p[0] == 0]
    diag_H1 = [p for p in diag if p[0] == 1]

    # plotting only the H1 persistence diagram
    fig, ax = plt.subplots(figsize=(4, 4))
    births = [b for _, (b, d) in diag_H1]
    deaths = [d if d != float("inf") else 1.0 for _, (b, d) in diag_H1]
    ax.scatter(births, deaths, color="blue", label="H1")

    ax.plot([0, 1.1], [0, 1.1], "k--", linewidth=1)

    ticks = np.arange(0, 1.1, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.tick_params(axis='both', labelsize=10)

    ax.set_xlabel("Birth (weight)", fontsize=10)
    ax.set_ylabel("Death (weight)", fontsize=10)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, 1.1])
    ax.set_title(title + " Persistence Diagram", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    if print_diagram:
        plt.show()
    else:
        plt.close()

    diag_H0_np = np.array([pair for _, pair in diag if _ == 0])
    diag_H1_np = np.array([pair for _, pair in diag if _ == 1])
    return {'dgms': [diag_H0_np, diag_H1_np]}



def analyze_persistence(G, pos, title):
    visualize_filtration(G, pos=pos, title=title)
    pos_high = plot_network(G, title)
    get_persistence_diagram(G, title)


def plot_clique(G, pos, clique_nodes):
    plt.figure(figsize=(10, 4))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightgray",
        edgecolors="black",
        linewidths=1.5,
        edge_color="gray",
        width=1.5,
        font_color="black",
    )

    # Highlight the clique
    clique_subgraph = G.subgraph(clique_nodes)
    nx.draw_networkx_nodes(
        clique_subgraph, pos, node_color="lightblue", edgecolors="blue", node_size=600
    )
    nx.draw_networkx_edges(clique_subgraph, pos, edge_color="blue", width=3)

    plt.title("Example of 3-Clique: {0, 1, 6}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_simplex(G, pos, clique_nodes):
    plt.figure(figsize=(10, 4))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightgray",
        edgecolors="black",
        linewidths=1.5,
        edge_color="gray",
        width=1.5,
        font_color="black",
    )

    # Highlight the 2-simplex (filled triangle)
    clique_pos = [pos[n] for n in clique_nodes]
    triangle = plt.Polygon(clique_pos, color="skyblue", alpha=0.5, zorder=0)
    plt.gca().add_patch(triangle)

    clique_subgraph = G.subgraph(clique_nodes)
    nx.draw_networkx_edges(clique_subgraph, pos, edge_color="blue", width=3)
    nx.draw_networkx_nodes(
        clique_subgraph, pos, node_color="lightblue", edgecolors="blue", node_size=600
    )

    plt.title("2-Simplex from Clique {0, 1, 6}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_flag_complex(G, fixed_position=None):
    pos = fixed_position if fixed_position is not None else nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 4))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightgray",
        edgecolors="black",
        linewidths=1.5,
        edge_color="gray",
        width=1.5,
        font_color="black",
    )

    cliques = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]

    for clique_nodes in cliques:
        clique_pos = [pos[n] for n in clique_nodes]
        triangle = plt.Polygon(clique_pos, color="skyblue", alpha=0.5, zorder=0)
        plt.gca().add_patch(triangle)

        clique_subgraph = G.subgraph(clique_nodes)
        nx.draw_networkx_edges(clique_subgraph, pos, edge_color="blue", width=2)
        nx.draw_networkx_nodes(
            clique_subgraph, pos, node_color="lightblue", edgecolors="blue", node_size=600
        )

    plt.title("Clique Complex of the network", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_complex_with_holes(G, pos=None, title="Clique complex of the network with weighted holes", min_cycle_length=4):

    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 4))
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    threshold = 0.0
    added_edges = [(u,v) for u,v,d in edges_sorted if d['weight'] >= threshold]
    subG = nx.Graph()
    subG.add_nodes_from(G.nodes)
    subG.add_edges_from(added_edges)
    
    # Build SimplexTree for homology computation
    nodes = list(G.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    st = SimplexTree()
    for node in nodes:
        st.insert([node_index[node]], filtration=0.0)
    
    for u, v, data in G.edges(data=True):
        filtration_val = 1 - data['weight']
        st.insert([node_index[u], node_index[v]], filtration=filtration_val)
    
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 3:
            i, j, k = [node_index[n] for n in clique]
            max_filt = max(1 - G[u][v]['weight'] 
                          for u,v in [(clique[0], clique[1]), (clique[1], clique[2]), (clique[0], clique[2])])
            st.insert(sorted([i,j,k]), filtration=max_filt)
    
    st.initialize_filtration()
    diag = st.persistence(homology_coeff_field=2)
    h1_generators = [(birth, death) for dim, (birth, death) in diag if dim == 1]
    
    nx.draw_networkx_nodes(subG, pos, node_size=300, node_color='lightgray', edgecolors='black')
    nx.draw_networkx_labels(subG, pos, font_size=10)
    nx.draw_networkx_edges(subG, pos, edgelist=added_edges, width=1.5, edge_color='dimgray')
    
    # Highlight 2-simplices (triangles)
    triangles = [c for c in nx.enumerate_all_cliques(subG) if len(c) == 3]
    for tri in triangles:
        tri_pos = [pos[n] for n in tri]
        poly = Polygon(tri_pos, closed=True, alpha=0.4, edgecolor='blue', facecolor='lightblue', zorder=0)
        plt.gca().add_patch(poly)
    
    # Highlight H1 generators (weighted holes)
    active_generators = 0
    cycles = nx.cycle_basis(subG)
    
    for birth, death in h1_generators:
        if birth <= 1.0 < death:  
            active_generators += 1
            candidate_cycles = [c for c in cycles if len(c) >= 4]
            if candidate_cycles:
                cycle = max(candidate_cycles, key=len)
                try:
                    cycle_pos = [pos[n] for n in cycle]
                    hull = ConvexHull(cycle_pos)
                    poly = Polygon([cycle_pos[i] for i in hull.vertices], closed=True,
                                 alpha=0.3, edgecolor='red', facecolor='lightcoral', zorder=0)
                    plt.gca().add_patch(poly)
                except:
                    poly = Polygon(cycle_pos, closed=True, alpha=0.3,
                                 edgecolor='red', facecolor='lightcoral', zorder=0)
                    plt.gca().add_patch(poly)
    
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', label='2-Simplices (Triangles)', alpha=0.4),
        Patch(facecolor='lightcoral', edgecolor='red', label='Weighted Holes (H1 generators)', alpha=0.3),
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

def randomize_weights(G, pos=None):
    G_sh = copy.deepcopy(G)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    random.shuffle(weights)

    for idx, (u, v) in enumerate(G_sh.edges()):
        G_sh[u][v]["weight"] = weights[idx]

    if pos is None:
        pos = nx.spring_layout(G_sh, seed=42, k=0.3, iterations=100)

    return G_sh, pos


def randomize_edges_and_weights(G, pos=None, nswap=None, max_tries=None):
    G_rnd = copy.deepcopy(G)

    # Apply edge swap while preserving degree sequence
    if nswap is None:
        nswap = 10 * G_rnd.number_of_edges()
    if max_tries is None:
        max_tries = nswap * 10

    G_rnd = nx.double_edge_swap(G_rnd, nswap=nswap, max_tries=max_tries, seed=42)

    original_weights = [G[u][v]["weight"] for u, v in G.edges()]
    random.shuffle(original_weights)

    for (u, v), w in zip(G_rnd.edges(), original_weights):
        G_rnd[u][v]["weight"] = w

    if pos is None:
        pos = nx.spring_layout(G_rnd, seed=42, k=0.3, iterations=100)

    return G_rnd, pos


def compute_hollowness(G, diagrams, homology_dim=1):
    """
    Compute
        - Hollowness h_k
        - Chain-length normalized hollowness tilde_h_k
    for H_k generators 
    """
    N = G.number_of_nodes()

    Hk_generators = diagrams['dgms'][homology_dim]
    Ngk = len(Hk_generators)

    if Ngk == 0:
        print(f"No generators found for H_{homology_dim}.")
        return 0.0, 0.0
    
    births = Hk_generators[:, 0]
    deaths = Hk_generators[:, 1]

    persistences = []
    length_gk = [] 

    for i in range(Ngk):
        if np.isfinite(births[i]) and np.isfinite(deaths[i]):
            persistences.append(deaths[i] - births[i])
        else:
            persistences.append(1 - births[i])

        G_sub = nx.Graph()
        for u, v, data in G.edges(data=True):
            if data['weight'] >= births[i]:
                G_sub.add_edge(u, v)

        cycles = nx.cycle_basis(G_sub)

        if len(cycles) > 0:
            shortest_cycle = min(cycles, key=len)
            length_gk.append(len(shortest_cycle))  
        else:
            length_gk.append(0)

    length_gk = np.array(length_gk)
    persistences = np.array(persistences)

    # Hollowness
    hk = np.mean(persistences)

    # Chain-length-normalized
    tilde_hk = np.mean((length_gk / N) * persistences)

    return hk, tilde_hk

