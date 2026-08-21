from training.scrpit_generate_agent import generate_agent
from metrics.Classe_RunResult import RunResult
from training.benchmark import train_agent_chrono
from metrics.Classe_EpochTimer import EpochTimer
from metrics.Classe_CommunicationCost import CommunicationCostComparator

# ──────────────────────────────────────────────────────────────
# Benchmark séquentiel
# ──────────────────────────────────────────────────────────────

def benchmark_sequential(
    communication_methods : dict,
    num_epochs            : int,
    graph,
    k                     : int,
    BATCH_SIZE            : int,
    N_AGENT               : int,
    DEVICE,
) -> tuple[list[RunResult], CommunicationCostComparator]:

    results    = []
    comparator = CommunicationCostComparator()   # ← 1 seule instance

    for method_name, communication_fn in communication_methods.items():
        print(f"\n{'='*60}\n  RUN — méthode : {method_name}\n{'='*60}")

        # ── Nouveaux agents pour chaque méthode ───────────────────────────
        agent_list = generate_agent(BATCH_SIZE, N_AGENT, DEVICE)

        # ── Entraînement + mesure ──────────────────────────────────────────
        metrics, timer, comm_cost = train_agent_chrono(
            agent_list, num_epochs, graph, k,
            communication_fn, method_name
        )

        # ── ✅ FIX 1 : alimenter le comparator avec comm_cost ─────────────
        comparator.add(comm_cost)

        # ── Résultat ──────────────────────────────────────────────────────
        result = RunResult(method_name, agent_list, metrics, timer, comm_cost)
        results.append(result)
        print(result)

        # ── Affichage du résumé de communication ──────────────────────────
        print(comm_cost.summary())

    # ── ✅ FIX 2 : rank() après que tous les comm_cost sont ajoutés ───────
    print("\n" + "═" * 60)
    print("  CLASSEMENT FINAL — comparator.rank()")
    print("═" * 60)
    comparator.rank()

    return results, comparator



"""
Runs multiple communication strategies sequentially, one after the other.

For each method, a FRESH set of agents is instantiated from scratch so
that results are never contaminated by prior runs. Each run uses
train_agent_chrono() internally, so timing data is always available.

Typical usage
-------------
    COMMUNICATION_METHODS = {
        "Average Global"       : avg_models_algorithm,
        "Hamiltonian Directed" : hamiltonian_directed_consensus,
    }

    results = benchmark_sequential(
        communication_methods = COMMUNICATION_METHODS,
        num_epochs            = 20,
        graph                 = graph,
        k                     = 5,
        BATCH_SIZE            = 32,
        N_AGENT               = 4,
        DEVICE                = torch.device("cuda"),
    )

Args:
    communication_methods : dict mapping a human-readable label (str) to a
                            communication function with signature
                            (agent_list, graph, k, epoch, num_epochs).
    num_epochs            : number of epochs per run.
    graph                 : NetworkX graph (directed or undirected).
    k                     : consensus iteration count.
    BATCH_SIZE            : batch size forwarded to generate_agent().
    N_AGENT               : number of agents per run.
    DEVICE                : torch.device used for model initialisation.

Returns:
    list[RunResult] : one RunResult per method, in insertion order.
                      Each RunResult exposes .metrics, .timer,
                      .val_accuracies, .val_losses, etc.
"""