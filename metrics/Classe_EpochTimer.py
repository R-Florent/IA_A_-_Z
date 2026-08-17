# metrics/Classe_EpochTimer.py

import time


class EpochTimer:
    """
    Chronomètre époque par époque.

    Usage correct :
        timer = EpochTimer()

        timer.start()          # ← démarre, ne retourne RIEN
        ... entraînement ...
        elapsed = timer.stop() # ← arrête ET retourne le temps écoulé

    Attributs publics :
        epoch_times      : liste des durées de chaque époque (secondes)
        cumulative_times : liste des temps cumulatifs (secondes)
    """

    def __init__(self):
        self.epoch_times      : list[float] = []
        self.cumulative_times : list[float] = []
        self._start_time      : float | None = None

    # ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        """
        Démarre le chrono au début d'une époque.
        Ne retourne RIEN — c'est stop() qui retourne le temps.

        CORRECTION : l'ancienne version était appelée avec
            elapsed = timer.start()
        ce qui donnait elapsed = None, puis elapsed était
        immédiatement écrasé par timer.stop(). La variable
        elapsed intermédiaire était donc toujours None et inutile.
        """
        if self._start_time is not None:
            # Sécurité : start() appelé deux fois sans stop()
            raise RuntimeError(
                "EpochTimer.start() appelé sans stop() préalable. "
                "Appelez stop() avant de relancer start()."
            )
        self._start_time = time.perf_counter()

    # ──────────────────────────────────────────────────────────────
    def stop(self) -> float:
        """
        Arrête le chrono, enregistre le temps et le cumulatif.

        Returns:
            float : durée de l'époque en secondes

        CORRECTION : c'est ICI qu'on récupère elapsed, pas dans start().
            elapsed = timer.stop()  ← correct
        """
        if self._start_time is None:
            raise RuntimeError(
                "EpochTimer.stop() appelé sans start() préalable."
            )

        elapsed          = time.perf_counter() - self._start_time
        self._start_time = None   # reset pour la prochaine époque

        self.epoch_times.append(elapsed)

        # Temps cumulatif = somme de toutes les époques précédentes + celle-ci
        previous_cumul = self.cumulative_times[-1] if self.cumulative_times else 0.0
        self.cumulative_times.append(previous_cumul + elapsed)

        return elapsed

    # ──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Remet le timer à zéro pour un nouveau run."""
        self.epoch_times      = []
        self.cumulative_times = []
        self._start_time      = None

    # ──────────────────────────────────────────────────────────────
    @property
    def total_time(self) -> float:
        """Temps total cumulatif du run (dernière valeur cumulative)."""
        return self.cumulative_times[-1] if self.cumulative_times else 0.0

    @property
    def mean_epoch_time(self) -> float:
        """Durée moyenne par époque."""
        return float(sum(self.epoch_times) / len(self.epoch_times)) \
               if self.epoch_times else 0.0

    @property
    def fastest_epoch(self) -> tuple[int, float]:
        """Retourne (index_époque, durée) de l'époque la plus rapide."""
        if not self.epoch_times:
            return -1, 0.0
        idx = int(min(range(len(self.epoch_times)),
                      key=lambda i: self.epoch_times[i]))
        return idx + 1, self.epoch_times[idx]   # +1 = numéro humain (1-based)

    @property
    def slowest_epoch(self) -> tuple[int, float]:
        """Retourne (index_époque, durée) de l'époque la plus lente."""
        if not self.epoch_times:
            return -1, 0.0
        idx = int(max(range(len(self.epoch_times)),
                      key=lambda i: self.epoch_times[i]))
        return idx + 1, self.epoch_times[idx]

    # ──────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        fast_i, fast_t = self.fastest_epoch
        slow_i, slow_t = self.slowest_epoch
        return (
            f"<EpochTimer "
            f"epochs={len(self.epoch_times)} | "
            f"total={self.total_time:.2f}s | "
            f"mean={self.mean_epoch_time:.2f}s/ep | "
            f"fastest=ep{fast_i}({fast_t:.2f}s) | "
            f"slowest=ep{slow_i}({slow_t:.2f}s)>"
        )