import torch
import threading
import copy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class ExperimentRunner:
    """
    Lance plusieurs expériences avec différentes topologies
    et compare les metrics
    """

    def __init__(self, num_epochs, num_agents, synchronization_func):
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.synchronization_func = synchronization_func
        self.results = {}  # {topology_name: metrics}

    # ============================================
    # 1️⃣ CRÉER LES TOPOLOGIES
    # ============================================

    def create_topologies(self):
        """
        Crée toutes les topologies à tester
        """
        topologies = {
            "Fully Connected": NetworkTopology.fully_connected_graph(self.num_agents),
            "Cycle": NetworkTopology.cycle_graph(self.num_agents),
            "Star": NetworkTopology.star_graph(self.num_agents),
            "Random (p=0.3)": NetworkTopology.random_graph(self.num_agents, p=0.3),
            "Random (p=0.5)": NetworkTopology.random_graph(self.num_agents, p=0.5),
            "Grid": NetworkTopology.grid_graph(self.num_agents),
        }
        return topologies

    # ============================================
    # 2️⃣ CRÉER LES AGENTS (FRAIS POUR CHAQUE EXP)
    # ============================================

    def create_fresh_agents(self, device='cpu'):
        """
        Crée une nouvelle liste d'agents vierges
        """
        agent_list = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i,
                model=YourModel(),  # ← Adapte selon ton modèle
                device=device
            )
            agent_list.append(agent)
        return agent_list

    # ============================================
    # 3️⃣ RUNNER PRINCIPAL
    # ============================================

    def run_experiment(self, topology_name, graph, agent_list, k=1):
        """
        Lance une expérience complète avec une topologie donnée
        """
        print(f"\n{'=' * 70}")
        print(f"🚀 EXPÉRIENCE: {topology_name}")
        print(f"{'=' * 70}\n")

        # Visualiser la topologie
        NetworkTopology.visualize_graph(graph, title=f"Network Topology: {topology_name}")

        # Initialiser les metrics
        metrics = ModelMetrics(agent_list)

        # ============================================
        # BOUCLE D'ENTRAÎNEMENT
        # ============================================

        for epoch in range(self.num_epochs):

            # 1️⃣ ENTRAÎNEMENT PARALLÈLE
            threads_list = []
            for agent in agent_list:
                thread = threading.Thread(
                    target=agent.train_and_validate,
                    args=(agent_list,)
                )
                threads_list.append(thread)

            # Start
            for thread in threads_list:
                thread.start()

            # Attendre la fin
            for thread in threads_list:
                thread.join()

            # 2️⃣ SYNCHRONISATION AVEC TOPOLOGIE
            self.synchronization_func(agent_list, graph, k)

            # 3️⃣ CALCULER LES DISTANCES
            distances = Agent.node_weight_metric(agent_list)
            for agent, distance in zip(agent_list, distances):
                agent.total_distance_list.append(distance)

            # 4️⃣ LOG DES METRICS
            metrics.log_metrics(epoch)

            # Affichage simple
            avg_distance = sum(distances) / len(distances)
            print(f"Epoch {epoch + 1}/{self.num_epochs} | Avg Distance: {avg_distance:.4f}")

        print(f"\n✅ Expérience '{topology_name}' terminée !\n")

        return metrics

    # ============================================
    # 4️⃣ LANCER TOUTES LES EXPÉRIENCES
    # ============================================

    def run_all_experiments(self, k=1, device='cpu'):
        """
        Lance TOUTES les topologies l'une après l'autre
        """
        topologies = self.create_topologies()

        for topology_name, graph in topologies.items():

            # Créer des agents frais pour chaque expérience
            agent_list = self.create_fresh_agents(device=device)

            # Ajouter les agents au graphe
            for agent_id, agent in enumerate(agent_list):
                graph.add_node(agent_id, agent=agent)

            # Lancer l'expérience
            metrics = self.run_experiment(
                topology_name=topology_name,
                graph=graph,
                agent_list=agent_list,
                k=k
            )

            # Sauvegarder les résultats
            self.results[topology_name] = {
                "metrics": metrics,
                "graph": graph,
                "agents": agent_list,
            }

        print("\n" + "=" * 70)
        print("🎉 TOUTES LES EXPÉRIENCES SONT TERMINÉES !")
        print("=" * 70 + "\n")

    # ============================================
    # 5️⃣ COMPARER LES RÉSULTATS
    # ============================================

    def compare_topologies(self):
        """
        Crée un rapport comparatif de toutes les topologies
        """
        print(f"\n{'=' * 70}")
        print(f"📊 COMPARAISON DES TOPOLOGIES")
        print(f"{'=' * 70}\n")

        comparison_data = {}

        for topology_name, result in self.results.items():
            metrics = result["metrics"]

            # Extraire les metrics clés
            variances = [m["variance"] for m in metrics.history]
            distances = [m["convergence"]["avg_distance"] for m in metrics.history]

            final_variance = variances[-1]
            final_distance = distances[-1]
            convergence_speed = variances[0] - final_variance  # Reduction

            comparison_data[topology_name] = {
                "final_variance": final_variance,
                "final_distance": final_distance,
                "convergence_speed": convergence_speed,
                "variance_trajectory": variances,
                "distance_trajectory": distances,
            }

            print(f"🔹 {topology_name}")
            print(f"   Final Variance:      {final_variance:.6f}")
            print(f"   Final Distance:      {final_distance:.4f}")
            print(f"   Convergence Speed:   {convergence_speed:.6f} (↓ est mieux)")
            print(f"   Num Edges:           {result['graph'].number_of_edges()}")
            print()

        return comparison_data

    # ============================================
    # 6️⃣ GRAPHIQUES DE COMPARAISON
    # ============================================

    def plot_comparison(self, comparison_data):
        """
        Crée des graphiques comparatifs
        """
        topology_names = list(comparison_data.keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ===== GRAPHIQUE 1 : Évolution de la variance =====
        ax = axes[0, 0]
        for topology_name, data in comparison_data.items():
            ax.plot(
                data["variance_trajectory"],
                marker='o',
                label=topology_name,
                linewidth=2,
                alpha=0.8
            )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Variance", fontsize=12)
        ax.set_title("Parameter Variance Over Time", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ===== GRAPHIQUE 2 : Évolution de la distance =====
        ax = axes[0, 1]
        for topology_name, data in comparison_data.items():
            ax.plot(
                data["distance_trajectory"],
                marker='s',
                label=topology_name,
                linewidth=2,
                alpha=0.8
            )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Avg Distance", fontsize=12)
        ax.set_title("Weight Distance Over Time", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ===== GRAPHIQUE 3 : Variance finale (bar chart) =====
        ax = axes[1, 0]
        variances = [comparison_data[t]["final_variance"] for t in topology_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(topology_names)))
        bars = ax.bar(topology_names, variances, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel("Final Variance", fontsize=12)
        ax.set_title("Final Parameter Variance Comparison", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Ajouter les valeurs au-dessus des barres
        for bar, val in zip(bars, variances):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        # ===== GRAPHIQUE 4 : Distance finale (bar chart) =====
        ax = axes[1, 1]
        distances = [comparison_data[t]["final_distance"] for t in topology_names]
        bars = ax.bar(topology_names, distances, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel("Final Distance", fontsize=12)
        ax.set_title("Final Weight Distance Comparison", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Ajouter les valeurs
        for bar, val in zip(bars, distances):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    # ============================================
    # 7️⃣ RAPPORT TEXTE COMPLET
    # ============================================

    def generate_report(self, comparison_data, output_file="experiment_report.txt"):
        """
        Génère un rapport texte complet
        """
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("RAPPORT D'EXPÉRIENCE - COMPARAISON DES TOPOLOGIES\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Nombre d'epochs: {self.num_epochs}\n")
            f.write(f"Nombre d'agents: {self.num_agents}\n\n")

            # Résultats pour chaque topologie
            for topology_name, data in comparison_data.items():
                f.write(f"\n{'─' * 70}\n")
                f.write(f"TOPOLOGIE: {topology_name}\n")
                f.write(f"{'─' * 70}\n")
                f.write(f"Final Variance:      {data['final_variance']:.6f}\n")
                f.write(f"Final Distance:      {data['final_distance']:.4f}\n")
                f.write(f"Convergence Speed:   {data['convergence_speed']:.6f}\n")
                f.write("\n")

            # Classement
            f.write(f"\n{'=' * 70}\n")
            f.write("CLASSEMENT PAR VARIANCE FINALE (↓ meilleur)\n")
            f.write(f"{'=' * 70}\n")

            sorted_by_var = sorted(
                comparison_data.items(),
                key=lambda x: x[1]["final_variance"]
            )

            for rank, (topology_name, data) in enumerate(sorted_by_var, 1):
                f.write(f"{rank}. {topology_name:<25} Variance: {data['final_variance']:.6f}\n")

            f.write("\n")

        print(f"✅ Rapport sauvegardé: {output_file}")