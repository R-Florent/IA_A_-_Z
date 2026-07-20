import copy
import matplotlib.pyplot as plt
import torch
from metrics.Classe_EpochTimer import EpochTimer

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def node_weight_metric(agent_list):

    avg_state_dict = copy.deepcopy(
        agent_list[0].model.state_dict()
    )

    # moyenne
    for key in avg_state_dict:

        for i in range(1, len(agent_list)):
            avg_state_dict[key] += (
                agent_list[i].model.state_dict()[key]
            )

        avg_state_dict[key] /= len(agent_list)

    # différences
    for agent_id, agent in enumerate(agent_list):

        print(f"\nAgent {agent_id}")

        total_distance = 0

        for key in avg_state_dict:

            diff = (
                avg_state_dict[key]
                - agent.model.state_dict()[key]
            )

            distance = torch.norm(diff).item()

            total_distance += distance

            print(
                f"{key:<30} "
                f"L2 distance = {distance:.6f}"
            )

        print(
            f"Distance totale = {total_distance:.6f}"
        )


def plot_all_agents_metrics(agent_list):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== LOSS =====
    for agent in agent_list:

        train_epochs = range(1, len(agent.train_losses) + 1)
        val_epochs = range(1, len(agent.loss_validate_list) + 1)

        axes[0].plot(
            train_epochs,
            agent.train_losses,
            label=f"Agent {agent.id} Train"
        )

        axes[0].plot(
            val_epochs,
            agent.loss_validate_list,
            linestyle='--',
            label=f"Agent {agent.id} Val"
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # ===== ACCURACY =====
    for agent in agent_list:

        train_epochs = range(1, len(agent.train_accuracies) + 1)
        val_epochs = range(1, len(agent.accuracy_validate_list) + 1)

        axes[1].plot(
            train_epochs,
            agent.train_accuracies,
            label=f"Agent {agent.id} Train"
        )

        axes[1].plot(
            val_epochs,
            agent.accuracy_validate_list,
            linestyle='--',
            label=f"Agent {agent.id} Val"
        )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_all_agent_node_weight_metric(agent_list):

    for agent in agent_list:
        plt.plot(
            agent.total_distance_list,
            label=f"Agent {agent.id}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Total Distance")
    plt.title("total_distance_node_weight")
    plt.legend()
    plt.show()

# ── Palette cohérente ──────────────────────────────────────────────────────
_AGENT_COLORS = plt.cm.tab10.colors   # jusqu'à 10 agents


def _base_fig(title: str, nrows=1, ncols=1, figsize=None):
    """Helper : crée une figure avec style uniforme."""
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (8 * ncols, 5 * nrows))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Graphe 1 — Std des poids par agent
# ──────────────────────────────────────────────────────────────────────────────

def plot_weight_std(agent_list: list, method_name: str = ""):
    """
    Trace la std des poids de chaque agent au fil des époques.
    Convergence = courbes qui se rapprochent ET descendent.
    """
    fig, ax = _base_fig(
        f"Std des poids par agent — {method_name}",
        figsize=(10, 5)
    )

    for i, agent in enumerate(agent_list):
        epochs = range(1, len(agent.weight_std_list) + 1)
        color  = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.plot(epochs, agent.weight_std_list,
                label=f"Agent {agent.id}",
                color=color,
                linewidth=1.8,
                marker="o", markersize=3)

    ax.set_xlabel("Époque")
    ax.set_ylabel("Std des poids")
    ax.set_title("Variance (std) interne des poids — convergence si les courbes se rejoignent")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Graphe 2 — Distance au modèle moyen par agent
# ──────────────────────────────────────────────────────────────────────────────

def plot_distance_to_mean(agent_list: list, method_name: str = ""):
    """
    Trace la distance L2 de chaque agent au modèle moyen.
    Convergence décentralisée = toutes les courbes → 0.
    """
    fig, ax = _base_fig(
        f"Distance au modèle moyen — {method_name}",
        figsize=(10, 5)
    )

    for i, agent in enumerate(agent_list):
        epochs = range(1, len(agent.total_distance_list) + 1)
        color  = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.plot(epochs, agent.total_distance_list,
                label=f"Agent {agent.id}",
                color=color,
                linewidth=1.8,
                marker="o", markersize=3)

    ax.set_xlabel("Époque")
    ax.set_ylabel("Distance L2 au modèle moyen")
    ax.set_title("Distance au consensus global — toutes → 0 = convergence parfaite")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Graphe 3 — Temps de calcul (par époque + cumulatif)
# ──────────────────────────────────────────────────────────────────────────────

def plot_compute_time(timer: EpochTimer, method_name: str = ""):
    """
    Deux sous-graphes :
      - Gauche  : temps par époque (bar chart)
      - Droite  : temps cumulatif (line chart)
    """
    epochs = range(1, len(timer.epoch_times) + 1)

    fig, (ax1, ax2) = _base_fig(
        f"Temps de calcul — {method_name}",
        nrows=1, ncols=2,
        figsize=(14, 5)
    )

    # ── Bar chart — temps par époque ──────────────────────────────
    bars = ax1.bar(epochs, timer.epoch_times,
                   color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Temps (s)")
    ax1.set_title("Temps par époque")
    ax1.grid(True, axis="y", alpha=0.3)

    # Annotation de la valeur sur chaque barre
    for bar, t in zip(bars, timer.epoch_times):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01 * max(timer.epoch_times),
                 f"{t:.1f}s",
                 ha="center", va="bottom", fontsize=7)

    # ── Line chart — temps cumulatif ──────────────────────────────
    ax2.plot(epochs, timer.cumulative_times,
             color="darkorange", linewidth=2,
             marker="o", markersize=4, label="Cumulatif")
    ax2.fill_between(epochs, timer.cumulative_times,
                     alpha=0.15, color="darkorange")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Temps cumulatif (s)")
    ax2.set_title("Temps cumulatif d'entraînement")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Graphe comparatif multi-méthodes (pour benchmark)
# ──────────────────────────────────────────────────────────────────────────────

def plot_benchmark_advanced(results: list):
    """
    Vue comparative des 3 métriques avancées pour tous les runs.
    `results` = liste de RunResult (voir benchmark.py).

    Chaque graphe superpose les courbes de l'agent 0 de chaque méthode.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Comparaison avancée des méthodes de communication", fontsize=14, fontweight="bold")

    method_colors = plt.cm.Set2.colors

    for i, result in enumerate(results):
        color      = method_colors[i % len(method_colors)]
        agent0     = result.agent_list[0]
        epochs_std = range(1, len(agent0.weight_std_list) + 1)
        epochs_dst = range(1, len(agent0.total_distance_list) + 1)
        epochs_time= range(1, len(result.timer.epoch_times) + 1)

        # ── Std poids ─────────────────────────────────────────────
        axes[0].plot(epochs_std, agent0.weight_std_list,
                     label=result.method_name, color=color,
                     linewidth=2, marker="o", markersize=3)

        # ── Distance au moyen ─────────────────────────────────────
        axes[1].plot(epochs_dst, agent0.total_distance_list,
                     label=result.method_name, color=color,
                     linewidth=2, marker="o", markersize=3)

        # ── Temps cumulatif ───────────────────────────────────────
        axes[2].plot(epochs_time, result.timer.cumulative_times,
                     label=result.method_name, color=color,
                     linewidth=2, marker="o", markersize=3)

    titles   = ["Std des poids (Agent 0)",
                "Distance au modèle moyen (Agent 0)",
                "Temps cumulatif (s)"]
    ylabels  = ["Std", "Distance L2", "Temps (s)"]

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Époque")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 1. STD DES POIDS — par agent, par époque
# ══════════════════════════════════════════════════════════════════════════════

def compute_weight_std(agent) -> float:
    """
    Calcule la std de tous les paramètres aplatis de l'agent.
    Un std faible + convergent entre agents = bonne convergence décentralisée.
    """
    all_params = []
    for param in agent.model.parameters():
        all_params.append(param.data.view(-1))          # aplatit chaque tenseur
    all_params_cat = torch.cat(all_params)               # vecteur unique
    return all_params_cat.std().item()


def log_weight_std(agent_list: list):
    """
    À appeler après chaque époque.
    Ajoute la std courante dans agent.weight_std_list pour chaque agent.
    """
    for agent in agent_list:
        std = compute_weight_std(agent)
        agent.weight_std_list.append(std)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTANCE AU MODÈLE MOYEN — par agent, par époque (déjà partiellement fait)
# ══════════════════════════════════════════════════════════════════════════════

def compute_distance_to_mean(agent_list: list) -> list[float]:
    """
    Calcule la distance L2 entre chaque agent et le modèle moyen global.
    Retourne une liste de distances (une par agent).

    C'est la métrique la plus révélatrice de la convergence décentralisée :
    - Si tous → 0 : consensus parfait
    - Si divergent : les agents gardent des solutions locales
    """
    # ── Calcul du modèle moyen ─────────────────────────────────────────────
    avg_state = copy.deepcopy(agent_list[0].model.state_dict())

    for key in avg_state:
        for i in range(1, len(agent_list)):
            avg_state[key] = avg_state[key] + agent_list[i].model.state_dict()[key]
        avg_state[key] = avg_state[key] / len(agent_list)

    # ── Distance L2 de chaque agent au modèle moyen ────────────────────────
    distances = []
    for agent in agent_list:
        total_dist = 0.0
        state = agent.model.state_dict()
        for key in avg_state:
            diff = avg_state[key] - state[key]
            total_dist += torch.norm(diff).item()
        distances.append(total_dist)

    return distances


def log_distance_to_mean(agent_list: list):
    """
    À appeler après chaque époque.
    Ajoute la distance courante dans agent.total_distance_list pour chaque agent.
    """
    distances = compute_distance_to_mean(agent_list)
    for agent, dist in zip(agent_list, distances):
        agent.total_distance_list.append(dist)