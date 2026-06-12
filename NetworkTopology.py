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
    def star_graph(n_agents):
        G = nx.star_graph(n_agents - 1)  # nx.star_graph(n) crée n+1 nœuds
        return G

    @staticmethod
    def random_graph(n_agents, p=0.3):
        G = nx.erdos_renyi_graph(n_agents, p)
        return G

    @staticmethod
    def grid_graph(n_agents):
        """
        Grille 2D (ex: 3×3 pour 9 agents)
        """
        # Trouver la grille la plus carrée possible
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


graph = NetworkTopology.fully_connected_graph(10)
NetworkTopology.visualize_graph(graph)