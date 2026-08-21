# metrics/Classe_CommunicationCost.py

import time

import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import Optional


from sympy.codegen.ast import none


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS — snapshot d'une époque
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpochCommunicationSnapshot:
    """
    Capture tous les coûts de communication d'une seule époque.

    Attributs :
        epoch              : numéro d'époque (0-based)
        n_messages         : nombre de transmissions agent→agent
        params_per_message : nombre de paramètres dans chaque message
        bytes_per_message  : taille en octets d'un message (float32 = 4 octets)
        total_bytes        : volume total échangé cette époque (octets)
        sync_time_ms       : durée réelle de la synchro (millisecondes)
        n_agents           : nombre d'agents impliqués
    """
    epoch              : int
    n_messages         : int
    params_per_message : int
    bytes_per_message  : float
    total_bytes        : float
    sync_time_ms       : float
    n_agents           : int
    source             : str

    @property
    def total_mb(self) -> float:
        """Volume total en mégaoctets."""
        return self.total_bytes / (1024 ** 2)

    @property
    def messages_per_agent(self) -> float:
        """Nombre moyen de messages envoyés par agent."""
        return self.n_messages / self.n_agents if self.n_agents > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class CommunicationCostMetric:
    """
    Mesure et compare le coût de communication entre méthodes de synchronisation.

    Métriques calculées :
    ┌──────────────────────┬────────────────────────────────────────────────┐
    │ Métrique             │ Description                                    │
    ├──────────────────────┼────────────────────────────────────────────────┤
    │ n_messages           │ Nb de transmissions agent→agent par époque     │
    │ params_per_message   │ Nb de paramètres envoyés par message           │
    │ total_bytes          │ Volume total échangé (octets)                  │
    │ sync_time_ms         │ Durée réelle de la synchronisation (ms)        │
    │ bytes_per_accuracy   │ Octets dépensés par point d'accuracy gagné     │
    │ efficiency_score     │ accuracy_gain / total_MB  (↑ = meilleur)       │
    └──────────────────────┴────────────────────────────────────────────────┘

    Usage :
        metric = CommunicationCostMetric(agent_list, method_name="Consensus")

        # Dans la boucle d'entraînement :
        with metric.measure_epoch(epoch):
            communication_fn(agent_list, graph, k)

        # Après le run :
        metric.summary()
        metric.plot()
    """

    def __init__(self, agent_list: list, method_name: str = "",graph = None):
        """
        Args:
            agent_list  : liste des agents (pour compter params et agents)
            method_name : nom de la méthode (pour les graphiques)
        """
        self.method_name  = method_name
        self.n_agents     = len(agent_list)
        self.graph        = graph

        # Compte les paramètres scalaires du modèle d'un agent
        self.n_params = int(sum(p.numel() for p in agent_list[0].model.parameters()))

        # float32 = 4 octets par paramètre
        self.bytes_per_param = 4

        # Historique époque par époque
        self.snapshots: list[EpochCommunicationSnapshot] = []

        # Accuracy finale (renseignée après le run via set_final_accuracy)
        self._final_accuracy: Optional[float] = None

#        self._epoch_n_messages: int | None = None

        self._epoch_n_messages : Optional[int] = None   # set par communication_fn
        self._last_source      : str = "fallback"       # traçabilité debu

    def set_epoch_messages(self, n_messages: int):
        """
        Appelé DEPUIS communication_fn pour déclarer le nombre
        réel de messages échangés pendant cette synchro.

        Exemple dans communication_fn :
            comm_cost.set_epoch_messages(n_agents * k)
        """
        print(f"[DEBUG] set_epoch_messages called → {n_messages} messages")
        self._epoch_n_messages = n_messages

    def _count_messages_from_graph(self) -> int:
        """
        Priorité de résolution du nombre de messages :
            1. set_epoch_messages() appelé par communication_fn  ← le plus précis
            2. Topologie du graphe NetworkX                      ← structurel
            3. Fallback : n_agents                               ← minimum garanti
        """
        # 1. Override explicite depuis communication_fn
        if self._epoch_n_messages is not None:
            return self._epoch_n_messages

        # 2. Depuis la topologie du graphe
        if self.graph is not None:
            if isinstance(self.graph, nx.DiGraph):
                return self.graph.number_of_edges()
            else:
                return self.graph.number_of_edges() * 2

        # 3. Fallback
        return self.n_agents

    def _record(self, epoch: int, sync_time_ms: float, num_epochs: int = 1):
        """
        Calcule et enregistre les métriques de coût pour une époque.
        n_messages est calculé depuis la topologie réelle du graphe.
        """
        n_messages = self._resolve_n_messages()
        print(f"[DEBUG] _record epoch={epoch} → n_messages={n_messages}")
        n_messages         = self._resolve_n_messages()
        params_per_message = self.n_params
        bytes_per_message  = params_per_message * self.bytes_per_param
        total_bytes        = n_messages * bytes_per_message

        snap = EpochCommunicationSnapshot(
            epoch              = epoch,
            n_messages         = n_messages,
            params_per_message = params_per_message,
            bytes_per_message  = bytes_per_message,
            total_bytes        = total_bytes,
            sync_time_ms       = sync_time_ms,
            n_agents           = self.n_agents,
            source             = self._last_source,
        )
        self.snapshots.append(snap)

        # Reset override pour l'époque suivante
        self._epoch_n_messages = None
    # ──────────────────────────────────────────────────────────────────────
    # CONTEXT MANAGER — mesure une époque
    # ──────────────────────────────────────────────────────────────────────

    class _EpochContext:
        """Context manager interne pour mesurer le temps de synchro."""

        def __init__(self, cost_metric: "CommunicationCostMetric", epoch: int,num_epochs: int):
            self._metric = cost_metric
            self._epoch  = epoch
            self._num_epochs = num_epochs
            self._t0     = None

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
            self._metric._record(self._epoch, elapsed_ms, self._num_epochs)

    def measure_epoch(self, epoch: int,num_epochs: int = 1) -> "_EpochContext":
        """
        Context manager : entoure le bloc de synchronisation.

        Exemple :
            with metric.measure_epoch(epoch):
                communication_fn(agent_list, graph, k)
        """
        return self._EpochContext(self, epoch,num_epochs)


    def override_n_messages(self, epoch_idx: int, n_messages: int):
        """
        Corrige le nombre de messages pour une époque donnée après enregistrement.
        Utile quand la topologie est connue après la mesure.

        Args:
            epoch_idx  : index dans self.snapshots
            n_messages : nombre réel de transmissions
        """
        snap = self.snapshots[epoch_idx]
        snap.n_messages   = n_messages
        snap.total_bytes  = n_messages * snap.bytes_per_message

    def set_final_accuracy(self, accuracy: float):
        """
        Renseigne l'accuracy finale pour calculer l'efficacité.

        Args:
            accuracy : accuracy finale en % (ex: 92.3)
        """
        self._final_accuracy = accuracy
    # ──────────────────────────────────────────────────────────────────────
    # RÉSOLUTION DU NOMBRE DE MESSAGES
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_n_messages(self) -> int:
        print(f"[DEBUG] _resolve_n_messages → _epoch_n_messages = {self._epoch_n_messages}")
        """
        Résout le nombre de messages selon la priorité :
            1. Valeur déclarée par set_epoch_messages()   ← précis
            2. Topologie du graphe NetworkX               ← structurel
            3. Fallback : n_agents                        ← minimum garanti
        """
        # ── Priorité 1 : override explicite depuis communication_fn ───────
        if self._epoch_n_messages is not None:
            return self._epoch_n_messages

        # ── Priorité 2 : topologie du graphe ─────────────────────────────
        if self.graph is not None:
            if isinstance(self.graph, nx.DiGraph):
                # Orienté : 1 message par arc
                return self.graph.number_of_edges()
            else:
                # Non orienté : 2 messages par arête (A→B et B→A)
                return self.graph.number_of_edges() * 2

        # ── Priorité 3 : fallback minimum ────────────────────────────────
        return self.n_agents

    # ──────────────────────────────────────────────────────────────────────
    # PROPRIÉTÉS AGRÉGÉES
    # ──────────────────────────────────────────────────────────────────────

    @property
    def total_bytes_all_epochs(self) -> float:
        """Volume total échangé sur tout le run (octets)."""
        return sum(s.total_bytes for s in self.snapshots)

    @property
    def total_mb_all_epochs(self) -> float:
        """Volume total en mégaoctets."""
        return self.total_bytes_all_epochs / (1024 ** 2)

    @property
    def total_messages_all_epochs(self) -> int:
        """Nombre total de transmissions sur tout le run."""
        return sum(s.n_messages for s in self.snapshots)

    @property
    def mean_sync_time_ms(self) -> float:
        """Durée moyenne de synchronisation par époque (ms)."""
        if not self.snapshots:
            return 0.0
        return float(np.mean([s.sync_time_ms for s in self.snapshots]))

    @property
    def total_sync_time_ms(self) -> float:
        """Durée totale cumulée de synchronisation (ms)."""
        return float(sum(s.sync_time_ms for s in self.snapshots))

    @property
    def efficiency_score(self) -> float:
        """
        Score d'efficacité = accuracy_finale / total_MB_échangés.

        Interprétation :
            ↑ élevé  → beaucoup d'accuracy gagnée pour peu de données échangées
            ↓ faible → méthode coûteuse en bande passante pour peu de gains

        Retourne 0.0 si accuracy non renseignée.
        """
        if self._final_accuracy is None or self.total_mb_all_epochs == 0:
            return 0.0
        return self._final_accuracy / self.total_mb_all_epochs

    @property
    def bytes_per_accuracy_point(self) -> float:
        """
        Octets dépensés par point d'accuracy gagné.

        Interprétation :
            ↓ faible → méthode efficiente (peu de données pour beaucoup d'accuracy)
            ↑ élevé  → méthode coûteuse

        Retourne inf si accuracy non renseignée.
        """
        if self._final_accuracy is None or self._final_accuracy == 0:
            return float("inf")
        return self.total_bytes_all_epochs / self._final_accuracy

    # ──────────────────────────────────────────────────────────────────────
    # RÉSUMÉ TEXTE
    # ──────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Affiche un résumé complet des coûts de communication."""
        acc_str  = f"{self._final_accuracy:.2f}%" \
                   if self._final_accuracy is not None else "N/A"
        eff_str  = f"{self.efficiency_score:.3f} acc/%/MB" \
                   if self._final_accuracy is not None else "N/A"
        bpa_str  = f"{self.bytes_per_accuracy_point/1024:.1f} KB/point" \
                   if self._final_accuracy is not None else "N/A"

        report = (
            f"\n{'═'*58}\n"
            f"  CommunicationCostMetric — {self.method_name}\n"
            f"{'─'*58}\n"
            f"  Agents              : {self.n_agents}\n"
            f"  Paramètres/modèle   : {self.n_params:,}\n"
            f"  Époques mesurées    : {len(self.snapshots)}\n"
            f"{'─'*58}\n"
            f"  VOLUME\n"
            f"    Octets/message    : {self.snapshots[0].bytes_per_message/1024:.1f} KB\n"
            f"    Total échangé     : {self.total_mb_all_epochs:.3f} MB\n"
            f"    Total messages    : {self.total_messages_all_epochs:,}\n"
            f"{'─'*58}\n"
            f"  TEMPS\n"
            f"    Synchro moy./ép.  : {self.mean_sync_time_ms:.2f} ms\n"
            f"    Synchro totale    : {self.total_sync_time_ms:.1f} ms\n"
            f"{'─'*58}\n"
            f"  EFFICACITÉ\n"
            f"    Accuracy finale   : {acc_str}\n"
            f"    Score efficacité  : {eff_str}\n"
            f"    Coût/point acc.   : {bpa_str}\n"
            f"{'═'*58}\n"
        )
        print(report)
        return report

    # ──────────────────────────────────────────────────────────────────────
    # GRAPHIQUES
    # ──────────────────────────────────────────────────────────────────────

    def plot(self):
        """
        Affiche 4 graphiques :
            [0,0] Volume échangé par époque (MB)
            [0,1] Temps de synchronisation par époque (ms)
            [1,0] Messages cumulatifs
            [1,1] Bytes cumulatifs (MB)
        """
        if not self.snapshots:
            print("[CommunicationCostMetric] Aucune donnée à afficher.")
            return

        epochs     = [s.epoch + 1 for s in self.snapshots]
        volumes_mb = [s.total_mb  for s in self.snapshots]
        times_ms   = [s.sync_time_ms for s in self.snapshots]
        cumul_msg  = np.cumsum([s.n_messages for s in self.snapshots])
        cumul_mb   = np.cumsum(volumes_mb)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(
            f"Coût de communication — {self.method_name}",
            fontsize=14, fontweight="bold"
        )

        # ── [0,0] Volume par époque ───────────────────────────────
        ax = axes[0, 0]
        ax.bar(epochs, volumes_mb, color="steelblue", edgecolor="black",
               alpha=0.8, width=0.6)
        ax.set_xlabel("Époque", fontsize=10)
        ax.set_ylabel("Volume échangé (MB)", fontsize=10)
        ax.set_title("Volume échangé par époque", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", alpha=0.3)

        # ── [0,1] Temps de synchronisation ───────────────────────
        ax = axes[0, 1]
        ax.plot(epochs, times_ms, marker="o", color="crimson",
                linewidth=2, markersize=6)
        ax.fill_between(epochs, times_ms, alpha=0.15, color="crimson")
        ax.axhline(self.mean_sync_time_ms, linestyle="--",
                   color="darkred", alpha=0.7,
                   label=f"Moyenne : {self.mean_sync_time_ms:.1f} ms")
        ax.set_xlabel("Époque", fontsize=10)
        ax.set_ylabel("Temps synchro (ms)", fontsize=10)
        ax.set_title("Durée synchronisation par époque", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── [1,0] Messages cumulatifs ─────────────────────────────
        ax = axes[1, 0]
        ax.step(epochs, cumul_msg, where="post", color="darkorange",
                linewidth=2.5)
        ax.fill_between(epochs, cumul_msg, step="post",
                        alpha=0.15, color="darkorange")
        ax.set_xlabel("Époque", fontsize=10)
        ax.set_ylabel("Messages cumulatifs", fontsize=10)
        ax.set_title("Transmissions cumulatives", fontsize=11,
                     fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))
        ax.grid(alpha=0.3)

        # ── [1,1] Bytes cumulatifs ────────────────────────────────
        ax = axes[1, 1]
        ax.plot(epochs, cumul_mb, marker="s", color="seagreen",
                linewidth=2, markersize=6)
        ax.fill_between(epochs, cumul_mb, alpha=0.15, color="seagreen")
        ax.set_xlabel("Époque", fontsize=10)
        ax.set_ylabel("Volume cumulatif (MB)", fontsize=10)
        ax.set_title("Volume cumulatif échangé", fontsize=11,
                     fontweight="bold")
        ax.grid(alpha=0.3)

        # Annotation accuracy si disponible
        if self._final_accuracy is not None:
            axes[1, 1].annotate(
                f"Accuracy finale : {self._final_accuracy:.1f}%\n"
                f"Efficacité : {self.efficiency_score:.2f} acc/MB",
                xy=(0.97, 0.05),
                xycoords="axes fraction",
                ha="right", va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#e8f5e9", edgecolor="#4caf50")
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# COMPARATEUR MULTI-MÉTHODES
# ══════════════════════════════════════════════════════════════════════════════

class CommunicationCostComparator:
    """
    Compare les coûts de communication entre plusieurs méthodes.

    Usage :
        comparator = CommunicationCostComparator()
        comparator.add(metric_consensus)
        comparator.add(metric_hamiltonian)
        comparator.plot_comparison()
        comparator.rank()
    """

    def __init__(self):
        self.metrics: list[CommunicationCostMetric] = []

    def add(self, metric: CommunicationCostMetric):
        """Ajoute une métrique de méthode au comparateur."""
        self.metrics.append(metric)

    def rank(self) -> list[tuple[str, float, float, float]]:
        """
        Classe les méthodes selon 3 critères :
            1. Volume total (↓ meilleur)
            2. Temps synchro moyen (↓ meilleur)
            3. Score efficacité (↑ meilleur)

        Returns:
            liste de tuples (method_name, total_mb, mean_ms, efficiency)
            triée par score efficacité décroissant
        """
        ranked = sorted(
            self.metrics,
            key=lambda m: m.efficiency_score,
            reverse=True   # ↑ efficacité = mieux
        )

        print(f"\n{'═'*70}")
        print(f"  CLASSEMENT DES MÉTHODES PAR EFFICACITÉ (accuracy / MB)")
        print(f"{'─'*70}")
        print(f"  {'Rang':<5} {'Méthode':<25} {'Total MB':<12} "
              f"{'Synchro moy':<14} {'Efficacité'}")
        print(f"{'─'*70}")

        results = []
        for rank, m in enumerate(ranked, 1):
            acc_str = f"{m._final_accuracy:.1f}%" \
                      if m._final_accuracy else "N/A"
            eff_str = f"{m.efficiency_score:.3f}" \
                      if m._final_accuracy else "N/A"
            print(f"  {rank:<5} {m.method_name:<25} "
                  f"{m.total_mb_all_epochs:<12.3f} "
                  f"{m.mean_sync_time_ms:<14.2f} "
                  f"{eff_str}")
            results.append((
                m.method_name,
                m.total_mb_all_epochs,
                m.mean_sync_time_ms,
                m.efficiency_score
            ))

        print(f"{'═'*70}\n")
        return results

    def plot_comparison(self):
        """
        Graphique comparatif multi-méthodes :
            [0,0] Volume total échangé (MB)       — barres
            [0,1] Temps synchro moyen (ms)        — barres
            [1,0] Score efficacité (acc/MB)       — barres horizontales
            [1,1] Radar cost vs accuracy          — scatter
        """
        if not self.metrics:
            print("[Comparator] Aucune métrique ajoutée.")
            return

        names    = [m.method_name        for m in self.metrics]
        volumes  = [m.total_mb_all_epochs for m in self.metrics]
        times    = [m.mean_sync_time_ms   for m in self.metrics]
        scores   = [m.efficiency_score    for m in self.metrics]
        accuracies = [
            m._final_accuracy if m._final_accuracy else 0.0
            for m in self.metrics
        ]

        colors  = plt.cm.tab10.colors
        x       = np.arange(len(names))
        width   = 0.5

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            "Comparaison du coût de communication entre méthodes",
            fontsize=14, fontweight="bold"
        )

        # ── [0,0] Volume total ────────────────────────────────────
        ax = axes[0, 0]
        bars = ax.bar(x, volumes, width, color=colors[:len(names)],
                      edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Volume total échangé (MB)", fontsize=10)
        ax.set_title("Volume total (↓ meilleur)", fontsize=11,
                     fontweight="bold")
        for bar, v in zip(bars, volumes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{v:.3f} MB", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # ── [0,1] Temps synchronisation ──────────────────────────
        ax = axes[0, 1]
        bars = ax.bar(x, times, width, color=colors[:len(names)],
                      edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Temps synchro moyen (ms)", fontsize=10)
        ax.set_title("Temps synchro moyen (↓ meilleur)", fontsize=11,
                     fontweight="bold")
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{t:.1f} ms", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # ── [1,0] Score efficacité ────────────────────────────────
        ax = axes[1, 0]
        sorted_idx = np.argsort(scores)[::-1]
        sorted_names  = [names[i]  for i in sorted_idx]
        sorted_scores = [scores[i] for i in sorted_idx]
        sorted_colors = [colors[i % len(colors)] for i in sorted_idx]

        hbars = ax.barh(
            range(len(sorted_names)), sorted_scores,
            color=sorted_colors, edgecolor="black", alpha=0.85
        )
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel("Score efficacité (accuracy / MB  ↑ meilleur)", fontsize=10)
        ax.set_title("Classement efficacité", fontsize=11, fontweight="bold")
        for bar, s in zip(hbars, sorted_scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{s:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # ── [1,1] Scatter coût vs accuracy ───────────────────────
        ax = axes[1, 1]
        for i, m in enumerate(self.metrics):
            acc = m._final_accuracy if m._final_accuracy else 0.0
            ax.scatter(
                m.total_mb_all_epochs, acc,
                s=200, color=colors[i % len(colors)],
                edgecolors="black", linewidths=1.5,
                zorder=3, label=m.method_name
            )
            ax.annotate(
                m.method_name,
                xy=(m.total_mb_all_epochs, acc),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9, fontweight="bold",
                color=colors[i % len(colors)]
            )

        # Zone idéale = coin haut-gauche (haute accuracy, faible coût)
        ax.axvline(
            np.mean(volumes), linestyle="--",
            color="gray", alpha=0.5, label="Coût moyen"
        )
        ax.axhline(
            np.mean(accuracies), linestyle="--",
            color="gray", alpha=0.5, label="Accuracy moyenne"
        )

        # Annotation zone idéale
        ax.text(
            min(volumes), max(accuracies),
            "✓ Zone idéale\n(faible coût, haute accuracy)",
            fontsize=8, color="green", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#e8f5e9", edgecolor="#4caf50", alpha=0.8)
        )

        ax.set_xlabel("Volume total échangé (MB)  ← meilleur", fontsize=10)
        ax.set_ylabel("Accuracy finale (%)  ↑ meilleur", fontsize=10)
        ax.set_title("Coût vs Accuracy\n(coin haut-gauche = méthode idéale)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()