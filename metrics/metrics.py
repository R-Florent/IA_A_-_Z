import copy
import matplotlib.pyplot as plt
import torch
from metrics.Classe_EpochTimer import EpochTimer

import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from agents.Classe_agent import Agent

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


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES INTERNES
# ══════════════════════════════════════════════════════════════════════════════

def _get_flat_weights(agent) -> np.ndarray:
    """
    Extrait et aplatit tous les paramètres d'un agent en un vecteur numpy 1D.
    C'est la représentation vectorielle du modèle utilisée par toutes les métriques.
    """
    parts = []
    for param in agent.model.parameters():
        parts.append(param.data.cpu().float().view(-1).numpy())
    return np.concatenate(parts)


def _get_all_flat_weights(agent_list) -> np.ndarray:
    """
    Retourne une matrice (n_agents × n_params) des poids aplatis.
    Chaque ligne = un agent.
    """
    return np.stack([_get_flat_weights(a) for a in agent_list])   # (N, D)


def _base_fig(title, nrows=1, ncols=1, figsize=None):
    """Helper : figure avec style uniforme."""
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (7 * ncols, 5 * nrows))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# 1. STD DES POIDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_weight_std(agent) -> float:
    """Std de tous les paramètres aplatis de l'agent."""
    vec = _get_flat_weights(agent)
    return float(np.std(vec))


def log_weight_std(agent_list: list):
    """À appeler après chaque époque. Enregistre la std dans l'agent."""
    for agent in agent_list:
        agent.weight_std_list.append(compute_weight_std(agent))


# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTANCE AU MODÈLE MOYEN
# ══════════════════════════════════════════════════════════════════════════════

def compute_distance_to_mean(agent_list: list) -> list[float]:
    """
    Distance L2 de chaque agent au modèle moyen global.
    → 0 pour tous = consensus parfait.
    """
    weights = _get_all_flat_weights(agent_list)    # (N, D)
    mean    = weights.mean(axis=0)                 # (D,)
    return [float(np.linalg.norm(w - mean)) for w in weights]


def log_distance_to_mean(agent_list: list):
    """À appeler après chaque époque."""
    distances = compute_distance_to_mean(agent_list)
    for agent, dist in zip(agent_list, distances):
        agent.total_distance_list.append(dist)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIMILARITÉ COSINUS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cosine_similarity_matrix(agent_list: list) -> np.ndarray:
    """
    Calcule la matrice de similarité cosinus (N × N) entre tous les agents.

    cos(A, B) = (A · B) / (||A|| × ||B||)

    Interprétation :
        1.0  → agents parfaitement alignés (mêmes poids)
        0.0  → poids orthogonaux
       -1.0  → poids opposés (rare en pratique)

    Args:
        agent_list: liste des agents

    Returns:
        np.ndarray (N × N) : matrice symétrique de similarités cosinus
    """
    weights = _get_all_flat_weights(agent_list)      # (N, D)
    n       = len(agent_list)

    # Normalisation L2 ligne par ligne
    norms        = np.linalg.norm(weights, axis=1, keepdims=True)   # (N, 1)
    norms        = np.where(norms == 0, 1e-10, norms)               # évite /0
    weights_norm = weights / norms                                   # (N, D)

    # Produit matriciel → matrice cosinus
    cosine_matrix = weights_norm @ weights_norm.T                    # (N, N)

    # Clip numérique pour rester dans [-1, 1]
    return np.clip(cosine_matrix, -1.0, 1.0)


def plot_cosine_similarity(agent_list: list, method_name: str = ""):
    """
    Affiche la heatmap de similarité cosinus entre tous les agents.

    Lecture :
        - Diagonale toujours à 1.0 (agent vs lui-même)
        - Hors-diagonale proche de 1 → agents très similaires → bon consensus
        - Valeurs faibles → agents divergents

    Args:
        agent_list  : liste des agents
        method_name : label pour le titre
    """
    n      = len(agent_list)
    matrix = compute_cosine_similarity_matrix(agent_list)
    labels = [f"A{a.id}" for a in agent_list]

    # Colormap : blanc=0.5, vert foncé=1.0, rouge=bas
    cmap = LinearSegmentedColormap.from_list(
        "cosine_cmap", ["#d73027", "#fee08b", "#ffffff", "#91cf60", "#1a9850"]
    )

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    fig.suptitle(f"Similarité Cosinus entre agents — {method_name}",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")

    # Annotations des valeurs dans chaque cellule
    for i in range(n):
        for j in range(n):
            val        = matrix[i, j]
            text_color = "black" if 0.2 < val < 0.9 else "white"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    plt.tight_layout()
    plt.show()

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# 4. HEATMAP DES DISTANCES L2
# ══════════════════════════════════════════════════════════════════════════════

def compute_distance_matrix(agent_list: list) -> np.ndarray:
    """
    Calcule la matrice des distances L2 (N × N) entre tous les couples d'agents.

    dist(A, B) = ||W_A - W_B||_2

    Interprétation :
        0   → agents identiques
        +∞  → agents très différents

    Args:
        agent_list: liste des agents

    Returns:
        np.ndarray (N × N) : matrice symétrique, diagonale = 0
    """
    weights = _get_all_flat_weights(agent_list)   # (N, D)
    n       = len(agent_list)
    matrix  = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = float(np.linalg.norm(weights[i] - weights[j]))

    return matrix


def plot_distance_heatmap(agent_list: list, method_name: str = ""):
    """
    Heatmap des distances L2 entre tous les couples d'agents.

    Lecture :
        - Diagonale toujours à 0 (agent vs lui-même)
        - Couleurs foncées hors-diagonale → agents proches → convergence
        - Couleurs claires → agents divergents

    Args:
        agent_list  : liste des agents
        method_name : label pour le titre
    """
    n      = len(agent_list)
    matrix = compute_distance_matrix(agent_list)
    labels = [f"A{a.id}" for a in agent_list]

    # Colormap : 0=vert foncé (proche), max=rouge (loin)
    cmap = LinearSegmentedColormap.from_list(
        "dist_cmap", ["#1a9850", "#91cf60", "#fee08b", "#d73027"]
    )

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    fig.suptitle(f"Heatmap des distances L2 entre agents — {method_name}",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(matrix, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, label="Distance L2")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")

    # Annotations
    max_val = matrix.max()
    for i in range(n):
        for j in range(n):
            val        = matrix[i, j]
            text_color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    plt.tight_layout()
    plt.show()

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# 5. PCA DES MODÈLES (réduction 2D)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_agents(agent_list: list, method_name: str = "",
                   epoch: int | None = None):
    """
    Projette les poids de chaque agent en 2D via PCA.

    Chaque modèle est un point dans un espace de très haute dimension (D).
    PCA réduit cet espace aux 2 directions de variance maximale.

    Lecture :
        - Points proches → agents similaires
        - Points éloignés → agents divergents
        - Cluster serré → convergence décentralisée réussie

    Note : PCA est linéaire → capture les tendances globales.
    Préférer t-SNE pour des structures non linéaires.

    Args:
        agent_list  : liste des agents
        method_name : label pour le titre
        epoch       : numéro d'époque (optionnel, pour le titre)
    """
    n       = len(agent_list)
    weights = _get_all_flat_weights(agent_list)   # (N, D)

    if n < 2:
        print("[PCA] Besoin d'au moins 2 agents.")
        return

    # Normalisation avant PCA
    scaler         = StandardScaler()
    weights_scaled = scaler.fit_transform(weights)

    n_components = min(2, n)
    pca          = PCA(n_components=n_components)
    coords       = pca.fit_transform(weights_scaled)   # (N, 2)

    explained    = pca.explained_variance_ratio_ * 100
    colors       = plt.cm.tab10.colors

    epoch_str = f" — Époque {epoch}" if epoch is not None else ""
    fig, ax   = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"PCA des modèles agents — {method_name}{epoch_str}",
                 fontsize=13, fontweight="bold")

    for i, agent in enumerate(agent_list):
        x, y = coords[i, 0], coords[i, 1] if n_components > 1 else 0.0
        color = colors[i % len(colors)]

        ax.scatter(x, y, s=200, color=color,
                   edgecolors="black", linewidths=1.5, zorder=3)
        ax.annotate(f"A{agent.id}",
                    xy=(x, y),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=11, fontweight="bold", color=color)

    # Centroïde = position du "consensus idéal"
    centroid = coords.mean(axis=0)
    ax.scatter(*centroid, s=300, color="black", marker="X",
               zorder=4, label="Centroïde (consensus)")

    # Cercle de dispersion autour du centroïde
    radius = float(np.max(np.linalg.norm(coords - centroid, axis=1)))
    circle = plt.Circle(centroid, radius, fill=False,
                        linestyle="--", color="gray", alpha=0.5)
    ax.add_patch(circle)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)" if n_components > 1
                  else "PC2", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 6. t-SNE DES MODÈLES (réduction 2D non linéaire)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne_agents(agent_list: list, method_name: str = "",
                    epoch: int | None = None,
                    perplexity: float | None = None):
    """
    Projette les poids de chaque agent en 2D via t-SNE.

    t-SNE préserve les structures locales (voisinages) contrairement à PCA.
    Particulièrement révélateur pour identifier des clusters d'agents
    qui ont convergé vers des solutions locales différentes.

    Lecture :
        - Cluster unique → convergence globale
        - Plusieurs clusters → sous-groupes d'agents divergents
        - Points isolés → agents outliers (problème de communication)

    Note : t-SNE est stochastique. Les positions absolues n'ont pas de sens,
    seules les distances RELATIVES entre points comptent.

    Args:
        agent_list  : liste des agents
        method_name : label pour le titre
        epoch       : numéro d'époque optionnel
        perplexity  : paramètre t-SNE (défaut : min(n-1, 5) — adaptatif)
    """
    n       = len(agent_list)
    weights = _get_all_flat_weights(agent_list)   # (N, D)

    if n < 3:
        print("[t-SNE] Besoin d'au moins 3 agents pour t-SNE. "
              "Utilisation de PCA à la place.")
        plot_pca_agents(agent_list, method_name, epoch)
        return

    # Perplexité adaptative : doit être < n
    perp = perplexity if perplexity is not None else min(n - 1, 5)

    # Normalisation
    scaler         = StandardScaler()
    weights_scaled = scaler.fit_transform(weights)

    # Réduction préalable par PCA si D >> N (accélère t-SNE)
    d = weights_scaled.shape[1]
    if d > 50:
        pca_pre        = PCA(n_components=min(50, n))
        weights_scaled = pca_pre.fit_transform(weights_scaled)

    tsne   = TSNE(n_components=2,
                  perplexity=perp,
                  random_state=42,
                  max_iter=1000,
                  init="pca")
    coords = tsne.fit_transform(weights_scaled)   # (N, 2)

    colors    = plt.cm.tab10.colors
    epoch_str = f" — Époque {epoch}" if epoch is not None else ""

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"t-SNE des modèles agents — {method_name}{epoch_str}",
                 fontsize=13, fontweight="bold")

    for i, agent in enumerate(agent_list):
        x, y  = coords[i, 0], coords[i, 1]
        color = colors[i % len(colors)]

        ax.scatter(x, y, s=200, color=color,
                   edgecolors="black", linewidths=1.5, zorder=3)
        ax.annotate(f"A{agent.id}",
                    xy=(x, y),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=11, fontweight="bold", color=color)

    # Centroïde
    centroid = coords.mean(axis=0)
    ax.scatter(*centroid, s=300, color="black", marker="X",
               zorder=4, label="Centroïde")

    # Score de dispersion (plus il est petit, plus les agents convergent)
    dispersion = float(np.mean(np.linalg.norm(coords - centroid, axis=1)))
    ax.set_title(f"Dispersion moyenne : {dispersion:.2f} "
                 f"(↓ = convergence)", fontsize=10)

    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 7. TABLEAU DE BORD COMPLET (toutes les métriques en une fois)
# ══════════════════════════════════════════════════════════════════════════════

def plot_full_metric_dashboard(agent_list: list, method_name: str = "",
                               epoch: int | None = None):
    """
    Affiche les 4 métriques spatiales en une seule figure 2×2 :
        [0,0] Similarité cosinus  |  [0,1] Heatmap distances L2
        [1,0] PCA 2D              |  [1,1] t-SNE 2D

    Idéal pour un snapshot de l'état des agents à la fin d'un run.

    Args:
        agent_list  : liste des agents
        method_name : nom de la méthode de communication
        epoch       : numéro d'époque (optionnel)
    """
    n         = len(agent_list)
    weights   = _get_all_flat_weights(agent_list)
    labels    = [f"A{a.id}" for a in agent_list]
    colors    = plt.cm.tab10.colors
    epoch_str = f" — Époque {epoch}" if epoch is not None else ""

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Tableau de bord métriques — {method_name}{epoch_str}",
        fontsize=15,
        fontweight="bold"
    )

    # ══════════════════════════════════════════════════════════════
    # [0,0] — Similarité Cosinus
    # ══════════════════════════════════════════════════════════════
    ax1      = fig.add_subplot(2, 2, 1)
    csim     = compute_cosine_similarity_matrix(agent_list)
    cmap_cos = LinearSegmentedColormap.from_list(
        "cos", ["#d73027", "#fee08b", "#ffffff", "#91cf60", "#1a9850"]
    )

    im1 = ax1.imshow(csim, cmap=cmap_cos, vmin=-1.0, vmax=1.0)
    plt.colorbar(im1, ax=ax1, label="Cosine Similarity", shrink=0.85)

    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Agent", fontsize=10)
    ax1.set_ylabel("Agent", fontsize=10)
    ax1.set_title("Similarité Cosinus\n(1.0 = identiques, ↑ = convergence)",
                  fontsize=11, fontweight="bold")

    # Annotation valeur dans chaque cellule
    for i in range(n):
        for j in range(n):
            v         = csim[i, j]
            textcolor = "black" if 0.2 < v < 0.85 else "white"
            ax1.text(j, i, f"{v:.3f}",
                     ha="center", va="center",
                     fontsize=9, color=textcolor, fontweight="bold")

    # ══════════════════════════════════════════════════════════════
    # [0,1] — Heatmap Distances L2
    # ══════════════════════════════════════════════════════════════
    ax2       = fig.add_subplot(2, 2, 2)
    dmat      = compute_distance_matrix(agent_list)
    cmap_dist = LinearSegmentedColormap.from_list(
        "dist", ["#1a9850", "#91cf60", "#fee08b", "#d73027"]
    )

    im2 = ax2.imshow(dmat, cmap=cmap_dist)
    plt.colorbar(im2, ax=ax2, label="Distance L2", shrink=0.85)

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Agent", fontsize=10)
    ax2.set_ylabel("Agent", fontsize=10)
    ax2.set_title("Heatmap Distances L2\n(0 = identiques, vert = proches)",
                  fontsize=11, fontweight="bold")

    # Annotation valeur dans chaque cellule
    max_dist = dmat.max() if dmat.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            v         = dmat[i, j]
            textcolor = "white" if v > max_dist * 0.6 else "black"
            ax2.text(j, i, f"{v:.1f}",
                     ha="center", va="center",
                     fontsize=9, color=textcolor, fontweight="bold")

    # ══════════════════════════════════════════════════════════════
    # [1,0] — PCA 2D
    # ══════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(2, 2, 3)

    # Normalisation + PCA
    scaler         = StandardScaler()
    weights_scaled = scaler.fit_transform(weights)        # (N, D)
    n_comp         = min(2, n)
    pca            = PCA(n_components=n_comp)
    pca_coords     = pca.fit_transform(weights_scaled)    # (N, 2)
    explained      = pca.explained_variance_ratio_ * 100

    centroid_pca   = pca_coords.mean(axis=0)
    pca_dispersion = float(np.mean(
        np.linalg.norm(pca_coords - centroid_pca, axis=1)
    ))

    # Cercle de dispersion
    radius = float(np.max(np.linalg.norm(pca_coords - centroid_pca, axis=1)))
    circle = plt.Circle(
        centroid_pca, radius,
        fill=False, linestyle="--", color="gray", alpha=0.4, linewidth=1.5
    )
    ax3.add_patch(circle)

    # Points agents
    for i, agent in enumerate(agent_list):
        x, y  = pca_coords[i, 0], pca_coords[i, 1] if n_comp > 1 else 0.0
        color = colors[i % len(colors)]
        ax3.scatter(x, y, s=200, color=color,
                    edgecolors="black", linewidths=1.5, zorder=3)
        ax3.annotate(
            f"A{agent.id}",
            xy=(x, y), xytext=(8, 8),
            textcoords="offset points",
            fontsize=11, fontweight="bold", color=color
        )

    # Centroïde
    ax3.scatter(*centroid_pca, s=280, color="black", marker="X",
                zorder=4, label=f"Centroïde\n(dispersion={pca_dispersion:.2f})")

    xlabel = f"PC1 ({explained[0]:.1f}% var.)"
    ylabel = f"PC2 ({explained[1]:.1f}% var.)" if n_comp > 1 else "PC2"
    ax3.set_xlabel(xlabel, fontsize=10)
    ax3.set_ylabel(ylabel, fontsize=10)
    ax3.set_title("PCA 2D des modèles\n(points proches = agents similaires)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, alpha=0.3)

    # ══════════════════════════════════════════════════════════════
    # [1,1] — t-SNE 2D
    # ══════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(2, 2, 4)

    if n >= 3:
        # Perplexité adaptative : doit être strictement < n
        perp = min(n - 1, 5)

        # Pré-réduction PCA si D >> n (stabilise + accélère t-SNE)
        w_in = weights_scaled.copy()
        if w_in.shape[1] > 50:
            w_in = PCA(
                n_components=min(50, n)
            ).fit_transform(w_in)

        tsne        = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            max_iter=1000,
            init="pca",
            learning_rate="auto"
        )
        tsne_coords   = tsne.fit_transform(w_in)          # (N, 2)
        centroid_tsne = tsne_coords.mean(axis=0)
        dispersion    = float(np.mean(
            np.linalg.norm(tsne_coords - centroid_tsne, axis=1)
        ))

        # Cercle de dispersion
        radius_tsne = float(
            np.max(np.linalg.norm(tsne_coords - centroid_tsne, axis=1))
        )
        circle_tsne = plt.Circle(
            centroid_tsne, radius_tsne,
            fill=False, linestyle="--", color="gray", alpha=0.4, linewidth=1.5
        )
        ax4.add_patch(circle_tsne)

        # Points agents
        for i, agent in enumerate(agent_list):
            x, y  = tsne_coords[i, 0], tsne_coords[i, 1]
            color = colors[i % len(colors)]
            ax4.scatter(x, y, s=200, color=color,
                        edgecolors="black", linewidths=1.5, zorder=3)
            ax4.annotate(
                f"A{agent.id}",
                xy=(x, y), xytext=(8, 8),
                textcoords="offset points",
                fontsize=11, fontweight="bold", color=color
            )

        # Centroïde
        ax4.scatter(*centroid_tsne, s=280, color="black", marker="X",
                    zorder=4,
                    label=f"Centroïde\n(dispersion={dispersion:.2f})")

        ax4.set_xlabel("t-SNE dim 1", fontsize=10)
        ax4.set_ylabel("t-SNE dim 2", fontsize=10)
        ax4.set_title(
            f"t-SNE 2D des modèles\n"
            f"(cluster unique = convergence | perplexité={perp})",
            fontsize=11, fontweight="bold"
        )
        ax4.legend(fontsize=8, loc="best")
        ax4.grid(True, alpha=0.3)

    else:
        # Fallback : pas assez d'agents pour t-SNE
        ax4.text(
            0.5, 0.5,
            "t-SNE nécessite\n≥ 3 agents\n\nUtilisez plot_pca_agents()\nà la place.",
            ha="center", va="center", fontsize=12,
            transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd",
                      edgecolor="#ffc107", linewidth=2)
        )
        ax4.set_title("t-SNE 2D (indisponible)", fontsize=11, fontweight="bold")
        ax4.axis("off")

    # ══════════════════════════════════════════════════════════════
    # Finalisation
    # ══════════════════════════════════════════════════════════════
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # laisse la place au suptitle
    plt.show()