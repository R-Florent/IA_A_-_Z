from training.scrpit_generate_agent import generate_agent
from metrics.Classe_RunResult import RunResult
from training.scprit_train_agent import train_agent

# ──────────────────────────────────────────────────────────────
# Benchmark séquentiel
# ──────────────────────────────────────────────────────────────

def train_sequential(
    communication_methods: dict,   # {"nom": fn, ...}
    num_epochs: int,
    graph,
    k: int,
    BATCH_SIZE: int,
    N_AGENT: int,
    DEVICE,
) -> list[RunResult]:
    """
    Entraîne chaque méthode l'une après l'autre.
    Les agents sont RÉINITIALISÉS entre chaque méthode pour une comparaison équitable.

    Args:
        communication_methods: dict {label: fn} — ex. {"average": synchronize_models_average}
        num_epochs:            nombre d'époques par run
        graph:                 topologie réseau (NetworkX graph)
        k:                     paramètre K pour consensu_algortyme
        BATCH_SIZE:            taille du batch
        N_AGENT:               nombre d'agents
        DEVICE:                torch.device

    Returns:
        Liste de RunResult (un par méthode)
    """
    results: list[RunResult] = []

    for method_name, communication_fn in communication_methods.items():
        print(f"\n{'='*60}")
        print(f"  RUN — méthode : {method_name}")
        print(f"{'='*60}")

        # ── Agents frais à chaque run ──────────────────────────────
        agent_list = generate_agent(BATCH_SIZE, N_AGENT, DEVICE)

        metrics = train_agent(agent_list, num_epochs, graph, k, communication_fn)

        result = RunResult(
            method_name=method_name,
            agent_list=agent_list,
            metrics=metrics,
        )

        results.append(result)
        print(result)

    return results