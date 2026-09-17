import random

import networkx as nx
import matplotlib.pyplot as plt

class NetworkTopology:

    @staticmethod
    def fully_connected_graph(n_agents):
        G = nx.complete_graph(n_agents)
        return G

    @staticmethod
    def cycle_graph(n_agents):
        G = nx.cycle_graph(n_agents)
        return G

    @staticmethod
    def cycle_graph_amiltionen(n_agents):
        if n_agents < 2:
            raise ValueError("Un graphe hamiltonien nécessite au moins 2 nœuds.")

        G = nx.DiGraph()
        G.add_nodes_from(range(n_agents))

        # ── Hamiltonien Cycle : 0→1→2→...→n-1→0 ────────
        for i in range(n_agents):
            G.add_edge(i, (i + 1) % n_agents)

        return G

    @staticmethod
    def star_graph(n_agents):
        G = nx.star_graph(n_agents - 1)  # nx.star_graph(n) crée n+1 nœuds
        return G

    @staticmethod
    def random_graph(n: int, edge_probability: float = 0.3):
        """
        Generate a random connected undirected graph.

        Properties:
            - No isolated nodes
            - The graph is connected
            - Each node can have between 1 and N-1 neighbors
            - Additional edges are randomly added
        """

        if n < 2:
            raise ValueError("n must be at least 2")

        # ----------------------------------------------------------
        # Step 1: Generate a random spanning tree
        # ----------------------------------------------------------
        graph = nx.random_labeled_tree(n)

        # ----------------------------------------------------------
        # Step 2: Add random edges
        # ----------------------------------------------------------
        for i in range(n):
            for j in range(i + 1, n):

                # Edge already exists in the spanning tree
                if graph.has_edge(i, j):
                    continue

                # Randomly add additional edges
                if random.random() < edge_probability:
                    graph.add_edge(i, j)

        return graph

    @staticmethod
    def bus_graph(n_agents):
        G = nx.Graph()
        G.add_nodes_from(range(n_agents))

        for i in range(n_agents - 1):
            G.add_edge(i, i + 1)

        return G

    @staticmethod
    def tree_graph(n_agents):
        if n_agents < 1:
            raise ValueError("error")

        G = nx.Graph()
        G.add_nodes_from(range(n_agents))

        # Arbre binaire : chaque nœud i a enfants 2i+1 et 2i+2
        for i in range(n_agents):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if left_child < n_agents:
                G.add_edge(i, left_child)

            if right_child < n_agents:
                G.add_edge(i, right_child)

        return G

    @staticmethod
    def directed_tree_graph(n_agents, direction="down"):

        G = nx.DiGraph()
        G.add_nodes_from(range(n_agents))

        for i in range(n_agents):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if direction == "down":
                if left_child < n_agents:
                    G.add_edge(i, left_child)
                if right_child < n_agents:
                    G.add_edge(i, right_child)

            elif direction == "up":
                if left_child < n_agents:
                    G.add_edge(left_child, i)
                if right_child < n_agents:
                    G.add_edge(right_child, i)

            else:
                raise ValueError("error")

        return G

    @staticmethod
    def grid_graph(n_agents):
        """
        Grille 2D (ex: 3×3 pour 9 agents)
        """
        cols = int(n_agents ** 0.5)
        while n_agents % cols != 0:
            cols -= 1
        rows = n_agents // cols
        G = nx.grid_2d_graph(rows, cols)
        # Renommer les nœuds (0, 1, 2...)
        return nx.convert_node_labels_to_integers(G)

    @staticmethod
    def visualize_graph(graph, title="Network Topology"):
        plt.figure(figsize=(8, 6))

        pos = nx.spring_layout(graph, seed=42)  # Positions fixes pour reproductibilité

        nx.draw_networkx_nodes(
            graph, pos,
            node_color='lightblue',
            node_size=500,
            edgecolors='black'
        )

        nx.draw_networkx_edges(
            graph, pos,
            width=2,
            alpha=0.6
        )

        nx.draw_networkx_labels(
            graph, pos,
            font_size=12,
            font_weight='bold'
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_neighbors_matrix(graph, n_agents):
        neighbors = {}
        for node in range(n_agents):
            neighbors[node] = list(graph.neighbors(node))
        return neighbors

    @staticmethod
    def get_neighbors(graph, agent_id):
        """Return list of neighbor IDs for a given agent"""
        return list(graph.neighbors(agent_id))

    @staticmethod
    def is_neighbor(graph, agent_id, other_agent_id):
        """Check if two agents are neighbors"""
        return other_agent_id in graph.neighbors(agent_id)