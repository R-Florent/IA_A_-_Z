import copy
import matplotlib.pyplot as plt
import torch
from datetime import datetime


class ModelMetrics:
    """
    Calcule les metrics importants pour l'entraînement distribué :
    - Variance des paramètres
    - Communication cost
    - Computation time
    - Parameter values statistics
    """

    def __init__(self, agent_list):
        self.agent_list = agent_list
        self.history = []

    # ============================================
    # 1️ VARIANCE DES PARAMÈTRES
    # ============================================

    @staticmethod
    def compute_parameter_variance(agent_list):
        """
        Calcule la variance des poids entre tous les agents
        Plus la variance est basse, plus les modèles sont synchronisés
        """
        variance_dict = {}

        # Récupère les state_dict de tous les agents
        all_state_dicts = [agent.model.state_dict() for agent in agent_list]

        # Pour chaque couche
        first_state = all_state_dicts[0]
        for key in first_state:
            # Collecte les tenseurs de tous les agents
            tensors_list = [state_dict[key] for state_dict in all_state_dicts]

            # Stack et calcule la variance
            stacked = torch.stack(tensors_list)  # [n_agents, ...]
            variance = torch.var(stacked, dim=0).mean().item()  # Moyenne de la variance

            variance_dict[key] = variance

        total_variance = sum(variance_dict.values())

        return variance_dict, total_variance

    # ============================================
    # 2️ DISTANCE ENTRE LES MODÈLES
    # ============================================

    @staticmethod
    def compute_weight_distances(agent_list):
        """
        Distance L2 entre chaque agent et la moyenne
        """
        # Calcule la moyenne
        avg_state_dict = copy.deepcopy(agent_list[0].model.state_dict())

        for key in avg_state_dict:
            for i in range(1, len(agent_list)):
                avg_state_dict[key] += agent_list[i].model.state_dict()[key]
            avg_state_dict[key] /= len(agent_list)

        # Distance de chaque agent par rapport à la moyenne
        distances = {}
        for agent_id, agent in enumerate(agent_list):
            total_distance = 0
            for key in avg_state_dict:
                diff = avg_state_dict[key] - agent.model.state_dict()[key]
                distance = torch.norm(diff).item()
                total_distance += distance

            distances[agent_id] = total_distance

        return distances, avg_state_dict

    # ============================================
    # 3️ COMMUNICATION COST
    # ============================================

    @staticmethod
    def compute_communication_cost(agent_list):
        """
        Estime la taille des données à communiquer
        """
        # Taille totale des paramètres d'un modèle
        total_params = 0
        param_count = 0

        state_dict = agent_list[0].model.state_dict()

        for key, param in state_dict.items():
            param_size = param.numel() * 4  # 4 bytes par float32
            total_params += param_size
            param_count += param.numel()

        # Coût de communication = taille × nombre d'agents
        # (Chaque agent envoie ses poids à tous les autres)
        communication_cost = {
            "total_params_bytes": total_params,
            "total_params_MB": total_params / (1024 ** 2),
            "total_params_count": param_count,
            "cost_per_communication": total_params,  # bytes envoyés par cycle
            "cost_per_cycle_all_agents": total_params * len(agent_list),  # Total envoyé
        }

        return communication_cost

    # ============================================
    # 4️ COMPUTATION TIME
    # ============================================

    @staticmethod
    def estimate_flops(agent_list):
        """
        Estime les FLOPs (opérations flottantes) du modèle
        Utilise l'input size pour calculer
        """
        model = agent_list[0].model

        # Input size (adapte selon ton dataset)
        input_size = (1, 3, 32, 32)  # CIFAR-10

        try:
            from thop import profile
            flops, params = profile(model, inputs=(torch.randn(input_size).to(next(model.parameters()).device),))
            return {
                "flops_per_forward": flops,
                "params": params,
                "flops_per_epoch": flops * 50000,  # Exemple : 50000 images
            }
        except ImportError:
            return {"error": "Installe 'thop' avec : pip install thop"}

    # ============================================
    # 5️ PARAMETER VALUES STATISTICS
    # ============================================

    @staticmethod
    def compute_parameter_statistics(agent_list):
        """
        Statistiques des valeurs des paramètres
        """
        stats = {}

        state_dict = agent_list[0].model.state_dict()

        for key, param in state_dict.items():
            param_flat = param.flatten()

            stats[key] = {
                "mean": param_flat.mean().item(),
                "std": param_flat.std().item(),
                "min": param_flat.min().item(),
                "max": param_flat.max().item(),
                "median": param_flat.median().item(),
                "sum": param_flat.sum().item(),
            }

        return stats

    # ============================================
    # 6️ GRADIENTS STATISTICS
    # ============================================

    @staticmethod
    def compute_gradient_statistics(agent_list):
        """
        Statistiques des gradients (pour analyser l'entraînement)
        """
        grad_stats = {}

        for agent_id, agent in enumerate(agent_list):
            agent_grads = {}

            for name, param in agent.model.named_parameters():
                if param.grad is not None:
                    grad_flat = param.grad.flatten()

                    agent_grads[name] = {
                        "mean": grad_flat.mean().item(),
                        "std": grad_flat.std().item(),
                        "norm": torch.norm(param.grad).item(),
                    }

            grad_stats[f"agent_{agent_id}"] = agent_grads

        return grad_stats

    # ============================================
    # 7️ CONVERGENCE METRICS
    # ============================================

    @staticmethod
    def compute_convergence_metrics(agent_list, epoch):
        """
        Métriques de convergence globale
        """
        distances, _ = ModelMetrics.compute_weight_distances(agent_list)
        variance_dict, total_variance = ModelMetrics.compute_parameter_variance(agent_list)

        return {
            "epoch": epoch,
            "max_distance": max(distances.values()),
            "min_distance": min(distances.values()),
            "avg_distance": sum(distances.values()) / len(distances),
            "total_variance": total_variance,
            "distances_per_agent": distances,
        }

    # ============================================
    # 8️ AFFICHAGE FORMATÉ
    # ============================================

    def print_all_metrics(self, epoch):
        """
        Affiche TOUTES les metrics de manière lisible
        """
        print(f"\n{'=' * 70}")
        print(f"📊 METRICS - Epoch {epoch}")
        print(f"{'=' * 70}\n")

        # Distances
        print("🔹 WEIGHT DISTANCES (Synchronisation)")
        print("-" * 70)
        distances, _ = self.compute_weight_distances(self.agent_list)
        for agent_id, dist in distances.items():
            bar = "█" * int(min(dist / 2, 40))
            print(f"  Agent {agent_id}: {dist:8.2f}  {bar}")
        avg_dist = sum(distances.values()) / len(distances)
        print(f"  Average:  {avg_dist:8.2f}\n")

        # Variance
        print("🔹 PARAMETER VARIANCE")
        print("-" * 70)
        var_dict, total_var = self.compute_parameter_variance(self.agent_list)
        for layer, var in var_dict.items():
            bar = "▓" * int(min(var * 100, 40))
            print(f"  {layer:<30} Var = {var:.6f}  {bar}")
        print(f"  Total Variance: {total_var:.6f}\n")

        # Communication Cost
        print("🔹 COMMUNICATION COST")
        print("-" * 70)
        comm_cost = self.compute_communication_cost(self.agent_list)
        print(f"  Total Parameters:        {comm_cost['total_params_count']:,}")
        print(f"  Model Size:              {comm_cost['total_params_MB']:.2f} MB")
        print(f"  Cost per Communication:  {comm_cost['cost_per_communication']:,} bytes")
        print(f"  Cost per Cycle (all):    {comm_cost['cost_per_cycle_all_agents']:,} bytes")
        print(f"  Cost per Cycle (all):    {comm_cost['cost_per_cycle_all_agents'] / (1024 ** 2):.2f} MB\n")

        # Parameter Statistics
        print("🔹 PARAMETER STATISTICS (Agent 0)")
        print("-" * 70)
        param_stats = self.compute_parameter_statistics(self.agent_list)
        for layer, stats in param_stats.items():
            print(f"  {layer:<30}")
            print(f"    Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f}")
            print(f"    Min:  {stats['min']:8.4f} | Max: {stats['max']:8.4f}\n")

        # Convergence
        print("🔹 CONVERGENCE STATUS")
        print("-" * 70)
        conv = self.compute_convergence_metrics(self.agent_list, epoch)
        print(f"  Max Distance:     {conv['max_distance']:.4f}")
        print(f"  Min Distance:     {conv['min_distance']:.4f}")
        print(f"  Avg Distance:     {conv['avg_distance']:.4f}")
        print(f"  Total Variance:   {conv['total_variance']:.6f}")

        print(f"\n{'=' * 70}\n")

    # ============================================
    # SAUVEGARDER L'HISTORIQUE
    # ============================================

    def log_metrics(self, epoch):
        """
        Enregistre les metrics dans l'historique
        """
        metrics = {
            "epoch": epoch,
            "timestamp": datetime.now(),
            "distances": self.compute_weight_distances(self.agent_list)[0],
            "variance": self.compute_parameter_variance(self.agent_list)[1],
            "convergence": self.compute_convergence_metrics(self.agent_list, epoch),
        }
        self.history.append(metrics)
        return metrics

    def plot_history(self):
        """
        Affiche l'évolution des metrics au fil des epochs
        """

        epochs = [m["epoch"] for m in self.history]
        variances = [m["variance"] for m in self.history]
        avg_distances = [m["convergence"]["avg_distance"] for m in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1 : Variance
        axes[0].plot(epochs, variances, marker='o', linewidth=2, label='Total Variance')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Variance")
        axes[0].set_title("Parameter Variance Over Time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 2 : Distance
        axes[1].plot(epochs, avg_distances, marker='s', linewidth=2, color='orange', label='Avg Distance')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Distance")
        axes[1].set_title("Weight Distance Over Time")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()