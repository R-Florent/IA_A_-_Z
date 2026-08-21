# =============================================================================
# training/train_pipeline.py
#
# PURPOSE
# -------
# Provides the core training loop used across all experiments.
# Three distinct functions are exposed, each serving a different use-case:
#
#   1. train_agent          — simple training loop, no timing
#   2. train_agent_chrono   — training loop WITH per-epoch timing (EpochTimer)
#   3. benchmark_sequential — runs multiple communication methods back-to-back,
#                             re-initialising fresh agents for each one so that
#                             comparisons are fair.
#
# All three share the SAME internal logic for:
#   - parallel agent training  (one thread per agent)
#   - inter-agent distance computation  (L2 to the mean model)
#   - weight synchronisation   (pluggable communication_fn)
#   - metric logging           (ModelMetrics)
#
# DEPENDENCIES
# ------------
#   agents.Classe_agent          → Agent, Agent.node_weight_metric()
#   metrics.Classe_model_metrics → ModelMetrics
#   metrics.Classe_EpochTimer    → EpochTimer
#   metrics.Classe_RunResult     → RunResult
#   training.script_generate_agent → generate_agent()
#
# USAGE
# -----
#   # Simple run
#   metrics = train_agent(agent_list, num_epochs, graph, k, comm_fn)
#
#   # Timed run
#   metrics, timer = train_agent_chrono(agent_list, num_epochs, graph, k, comm_fn)
#
#   # Full benchmark
#   results = benchmark_sequential(COMMUNICATION_METHODS, num_epochs,
#                                  graph, k, BATCH_SIZE, N_AGENT, DEVICE)
# =============================================================================

import threading

from agents.Classe_agent import Agent
from metrics.Classe_model_metrics import ModelMetrics
from metrics.Classe_EpochTimer import EpochTimer
from metrics.Classe_RunResult import RunResult
from training.scrpit_generate_agent import generate_agent
from metrics.Classe_CommunicationCost import CommunicationCostComparator, CommunicationCostMetric


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_parallel_training(agent_list: list, epoch: int, num_epochs: int) -> None:
    """
    Launches one training thread per agent and waits for all to finish.

    Each thread calls agent.train_and_validate(agent_list), which performs
    one epoch of local SGD and validation on the agent's private dataset.

    Threading is used here because agents train on independent data shards;
    there is no shared-state mutation during the forward/backward pass, so
    a simple thread-per-agent pattern is safe.

    Args:
        agent_list : list of Agent objects to train in parallel.
        epoch      : current epoch index (0-based), used only for logging.
        num_epochs : total number of epochs, used only for logging.
    """
    threads = [
        threading.Thread(target=agent.train_and_validate, args=[agent_list])
        for agent in agent_list
    ]

    for t in threads:
        t.start()

    print(f"  [Epoch {epoch + 1}/{num_epochs}] threads started …")

    for t in threads:
        t.join()


def _log_inter_agent_distances(agent_list: list) -> None:
    """
    Computes the L2 distance from each agent's model to the mean model,
    then appends the result to each agent's total_distance_list.

    This is called BEFORE weight synchronisation so that the distance
    reflects the natural divergence produced by local training — not
    the post-communication state.

    Args:
        agent_list : list of Agent objects.
    """
    distances = Agent.node_weight_metric(agent_list)
    for agent, dist in zip(agent_list, distances):
        agent.total_distance_list.append(dist)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMPLE TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(
    agent_list     : list,
    num_epochs     : int,
    graph,
    k              : int,
    communication_fn,
) -> ModelMetrics:
    """
    Trains all agents for num_epochs using the given communication strategy.
    No timing is performed — use train_agent_chrono() if you need per-epoch
    wall-clock measurements.

    Training loop per epoch
    -----------------------
    1. Parallel local training  (one thread per agent)
    2. Inter-agent distance logging  (L2 to mean model)
    3. Weight synchronisation   (communication_fn)
    4. Metric logging           (loss, accuracy, distance)

    Args:
        agent_list       : list of initialised Agent objects.
        num_epochs       : number of training epochs.
        graph            : NetworkX graph encoding the communication topology.
        k                : number of consensus iterations passed to
                           communication_fn.
        communication_fn : callable with signature
                           (agent_list, graph, k, epoch, num_epochs) -> None

    Returns:
        ModelMetrics : fully logged metrics object for all agents.
    """
    metrics = ModelMetrics(agent_list)

    for epoch in range(num_epochs):
        _run_parallel_training(agent_list, epoch, num_epochs)
        _log_inter_agent_distances(agent_list)
        communication_fn(agent_list, graph, k, epoch, num_epochs)
        metrics.log_metrics(epoch)

    print("Training complete.")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. TIMED TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_agent_chrono(
    agent_list     : list,
    num_epochs     : int,
    graph,
    k              : int,
    communication_fn,
    method_name     : str = "",
    ) -> tuple[ModelMetrics, EpochTimer, CommunicationCostMetric]:
    """
    Identical to train_agent() but wraps each epoch with an EpochTimer so
    that per-epoch wall-clock durations are recorded and later available for
    plotting or comparison.

    Timer scope per epoch
    ---------------------
    start()  ← before parallel training
        - parallel training
        - distance logging
        - weight synchronisation
        - metric logging
    stop()   ← after everything → captures the FULL epoch cost

    Why stop() after metrics.log_metrics()?
        We want the timer to represent the true cost of a full training step,
        including any overhead from synchronisation and logging, so that the
        benchmark reflects real-world wall-clock behaviour.

    Args:
        agent_list       : list of initialised Agent objects.
        num_epochs       : number of training epochs.
        graph            : NetworkX graph encoding the communication topology.
        k                : number of consensus iterations.
        communication_fn : callable (agent_list, graph, k, epoch, num_epochs)

    Returns:
        (ModelMetrics, EpochTimer)
            ModelMetrics : logged metrics for all agents.
            EpochTimer   : recorded epoch durations and cumulative times.
    """
    metrics = ModelMetrics(agent_list)
    timer   = EpochTimer()
    comm_cost = CommunicationCostMetric(agent_list, method_name,graph)

    for epoch in range(num_epochs):

        # ── Start timer — covers the ENTIRE epoch ─────────────────
        timer.start()

        _run_parallel_training(agent_list, epoch, num_epochs)
        _log_inter_agent_distances(agent_list)

        with comm_cost.measure_epoch(epoch, num_epochs):
            communication_fn(
                agent_list,
                graph,
                k,
                epoch,
                num_epochs,
                comm_cost=comm_cost
            )


        metrics.log_metrics(epoch)

        # ── Stop timer — returns elapsed seconds for this epoch ───
        elapsed = timer.stop()

        # ── Synchronisation + mesure du coût ──────────────────────
        # NOUVEAU : on entoure uniquement la communication
        # pour mesurer son coût isolément (pas le train)

        print(f"  [Epoch {epoch + 1}/{num_epochs}] ✓  "
              f"time={elapsed:.2f}s  "
              f"cumulative={timer.total_time:.2f}s")

    print(f"\n{'─' * 50}")
    print(f"  Training complete — {timer}")
    print(f"{'─' * 50}\n")

    return metrics, timer , comm_cost
