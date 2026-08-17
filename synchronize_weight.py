import copy
from agents.Classe_agent import Agent
from topologies.NetworkTopology import NetworkTopology
from hyperparametres import NUM_EPOCHES


def Sequential_AC(agent_list, graph):
    """
    Sequential Pairwise Averaging (Sequential Gossip Protocol).

    Each agent iterates over its neighbors one by one and averages
    its weights with each neighbor sequentially. This is an asymmetric
    update: only the *calling* agent updates its weights, not the neighbor.

    Mathematical Foundation
    -----------------------
    This is a sequential variant of the **Gossip / Pairwise Averaging**
    protocol (Kempe et al., 2003 ; Boyd et al., 2006).

    In its symmetric form, given two agents i and j:

        w_i ← (w_i + w_j) / 2
        w_j ← (w_i + w_j) / 2   ← NOT done here (asymmetric)

    This asymmetric version does NOT preserve the global weight sum,
    which means it is NOT equivalent to a doubly stochastic mixing step.
    It can still converge, but convergence is not theoretically guaranteed
    to the true global average.

    ⚠️ Note on asymmetry
    --------------------
    Because only `agent` is updated (not `neighbor`), the mixing matrix
    is row-stochastic but not column-stochastic. This breaks the
    sum-preserving property of average consensus. Consider using
    `consensus_step` for a symmetric, theoretically grounded update.

    Reference
    ---------
    - Boyd, S. et al. (2006). "Randomized gossip algorithms."
      IEEE Transactions on Information Theory, 52(6), 2508–2530.
    - Kempe, D., Dobra, A., & Gehrke, J. (2003). "Gossip-based computation
      of aggregate information." FOCS 2003.

    Parameters
    ----------
    agent_list : list[Agent]
        List of all agents in the federated network. Each agent must have
        an `.id` attribute and a `.model` PyTorch nn.Module.
    graph : networkx.Graph
        The communication topology. An edge (i, j) means agent i and
        agent j can communicate directly.

    Returns
    -------
    None
        Agents' model weights are updated in-place.

    Complexity
    ----------
    O(N * D * avg_degree) where N = number of agents, D = number of
    parameters per model.
    """
    for agent in agent_list:
        neighbors_ids = NetworkTopology.get_neighbors(graph, agent.id)

        for neighbor_id in neighbors_ids:
            neighbor = agent_list[neighbor_id]
            # Average weights between agent and one neighbor
            my_weights = agent.model.state_dict()
            neighbor_weights = neighbor.model.state_dict()

            averaged = {
                key: (my_weights[key] + neighbor_weights[key]) / 2
                for key in my_weights
            }
            # ⚠️ Only agent is updated — neighbor keeps its old weights
            agent.model.load_state_dict(averaged)


def consensus_step(agent_list, graph):
    """
    Synchronous Local Averaging — One Step of Average Consensus.

    Each agent simultaneously computes the mean of its own weights
    and the weights of all its direct neighbors, then all agents
    update at the same time.

    Mathematical Foundation
    -----------------------
    This implements one iteration of the **Linear Average Consensus**
    algorithm (Xiao & Boyd, 2004 ; Olfati-Saber et al., 2007):

        w_i(t+1) = Σ_{j ∈ N_i ∪ {i}}  (1 / (1 + |N_i|)) * w_j(t)

    In matrix form across all agents:

        W(t+1) = P · W(t)

    where P is the **mixing matrix**. Here, P_ij = 1 / deg(i)  if j ∈ N_i,
    and 0 otherwise. This matrix is **row-stochastic** by construction.

    For convergence to the true global average, P must also be
    **column-stochastic** (i.e., doubly stochastic). This holds
    automatically for regular graphs (all nodes have the same degree),
    e.g., rings, grids, or complete graphs.

    The simultaneous update (computing all new_weights before applying
    any of them) is critical — it ensures the algorithm is synchronous
    and mathematically consistent with the matrix formulation.

    Convergence
    -----------
    Under repeated application, all agents converge to the global average:

        lim_{t→∞} w_i(t) = (1/N) * Σ_i w_i(0)   for all i

    The rate of convergence is governed by the spectral gap:

        ρ = |λ₂(P)|  (second largest eigenvalue of P)

    A smaller ρ means faster convergence. The number of steps needed
    for ε-convergence scales as O(log(1/ε) / log(1/ρ)).

    Reference
    ---------
    - Xiao, L., & Boyd, S. (2004). "Fast linear iterations for distributed
      averaging." Systems & Control Letters, 53(1), 65–78.
    - Olfati-Saber, R., Fax, J.A., & Murray, R.M. (2007). "Consensus and
      cooperation in networked multi-agent systems." Proceedings of the IEEE.

    Parameters
    ----------
    agent_list : list[Agent]
        List of all agents. Each agent must have an `.id` attribute and
        a `.model` PyTorch nn.Module.
    graph : networkx.Graph
        The communication topology graph.

    Returns
    -------
    None
        All agents' model weights are updated in-place, simultaneously.

    Note
    ----
    The two-phase design (compute all new_weights FIRST, then apply them)
    is essential. Updating agents one by one would cause later agents to
    average with already-updated neighbors, breaking the synchronous
    guarantee.
    """
    new_weights = {}

    # ── Phase 1 : compute all new weight vectors (no model is touched yet)
    for agent in agent_list:
        neighbors = NetworkTopology.get_neighbors(graph, agent.id)
        current = agent.model.state_dict()

        # Start from own weights, then accumulate neighbors
        avg = {k: current[k].clone() for k in current}
        count = 1

        for n in neighbors:
            neighbor_weights = agent_list[n].model.state_dict()
            for k in avg:
                avg[k] += neighbor_weights[k]
            count += 1

        for k in avg:
            avg[k] /= count

        new_weights[agent.id] = avg

    # ── Phase 2 : simultaneous update (all agents at once)
    for agent in agent_list:
        agent.model.load_state_dict(new_weights[agent.id])


def Average_consensus_algorithm(agent_list, graph, K=5):
    """
    Iterative Average Consensus over K rounds (Synchronous Gossip).

    Repeatedly applies the local averaging step (`consensus_step` logic)
    for K iterations, progressively driving all agents toward the global
    average of their initial weights. Convergence is monitored after
    each iteration via the L2 weight distance metric.

    Mathematical Foundation
    -----------------------
    This is the **synchronous average consensus algorithm** of
    DeGroot (1974) and Tsitsiklis (1984), generalized to vector-valued
    states (model weight tensors):

        r_i(k+1) = Σ_{j ∈ N_i ∪ {i}}  (1 / |N_i ∪ {i}|) · r_j(k)

    In matrix form:

        R(k) = P^k · R(0)

    where R(0) is the matrix of initial weight vectors (one row per agent)
    and P is the mixing matrix. After K steps:

        R(K) → (1/N) · 1·1ᵀ · R(0)   as K → ∞

    meaning every row of R(K) converges to the true global average.

    Convergence rate per iteration: the error decays as ρ^k where
    ρ = |λ₂(P)| < 1 (spectral gap condition).

    The printed `avg_distance` after each round is the mean L2 deviation
    of agent weights from the global mean — a direct empirical measure
    of consensus progress.

    Reference
    ---------
    - DeGroot, M.H. (1974). "Reaching a consensus." Journal of the
      American Statistical Association, 69(345), 118–121.
    - Tsitsiklis, J.N. (1984). "Problems in decentralized decision making
      and computation." Ph.D. thesis, MIT.
    - Sun, T., Li, D., & Wang, B. (2021). "Decentralized Federated
      Averaging." arXiv:2104.11375.

    Parameters
    ----------
    agent_list : list[Agent]
        List of all agents in the network.
    graph : networkx.Graph
        Communication topology. Convergence speed depends heavily on
        graph connectivity (spectral gap of the adjacency/mixing matrix).
    K : int, optional
        Number of consensus iterations. Default is 5.
        More iterations → closer to global average, but more communication.

    Returns
    -------
    r : dict[int, dict[str, torch.Tensor]]
        Final averaged state dicts, keyed by agent id.
        Each value is a state dict (layer_name → weight tensor).

    Printed output
    --------------
    After each of the K iterations:
        → avrg distance after iteration k: X.XXXXXX
    A decreasing value indicates the agents are converging toward consensus.

    Complexity
    ----------
    O(K * N * D * avg_degree) where:
        K          = number of iterations
        N          = number of agents
        D          = total number of model parameters
        avg_degree = average number of neighbors per agent
    """
    # Snapshot initial weights
    r = {
        agent_id: copy.deepcopy(agent.model.state_dict())
        for agent_id, agent in enumerate(agent_list)
    }

    for k in range(K):

        r_new = {}

        for agent_id, agent in enumerate(agent_list):
            # Include self + all neighbors in the local average
            participants = [agent.id] + list(graph.neighbors(agent.id))

            avg_state = copy.deepcopy(
                agent_list[participants[0]].model.state_dict()
            )

            for key in avg_state:
                for participant_id in participants[1:]:
                    avg_state[key] += r[participant_id][key]
                avg_state[key] /= len(participants)

            r_new[agent_id] = avg_state

        r = r_new

        # Apply new weights to all agents
        for agent_id, agent in enumerate(agent_list):
            agent.model.load_state_dict(r[agent_id])

        # Monitor convergence
        distances = Agent.node_weight_metric(agent_list)
        avg_distance = sum(distances) / len(distances)
        print(f"    → avrg distance after iteration {k+1}: {avg_distance:.6f}")

    return r


def avg_models_algorithm(agent_list):
    """
    Centralized Federated Averaging (FedAvg — Global Aggregation Step).

    Computes the exact arithmetic mean of all agents' model weights
    and broadcasts the result to every agent. This is the canonical
    server-side aggregation step of Federated Learning.

    Mathematical Foundation
    -----------------------
    This implements the **FedAvg** aggregation rule
    (McMahan et al., 2017), assuming equal weight for every agent
    (uniform aggregation):

        w_global = (1/N) * Σ_{i=1}^{N} w_i

    where w_i is the full state dict (weight tensor collection) of agent i.

    This is the **optimal one-shot consensus** solution: it reaches the
    exact global average in a single round, but requires a central
    coordinator (server) that can access all agents simultaneously.
    It is therefore NOT decentralized.

    Compared to iterative consensus (e.g., `Average_consensus_algorithm`),
    this is O(1) in rounds but requires centralized communication,
    making it unsuitable for fully peer-to-peer settings.

    Reference
    ---------
    - McMahan, B. et al. (2017). "Communication-efficient learning of
      deep networks from decentralized data." AISTATS 2017.
      (The original FedAvg paper.)

    Parameters
    ----------
    agent_list : list[Agent]
        List of all agents. Each agent must expose a `.model` attribute
        (PyTorch nn.Module).

    Returns
    -------
    None
        All agents' model weights are updated in-place to the global
        average. After this call, every agent has strictly identical weights.

    Note
    ----
    This function assumes all agents contribute equally (weight = 1/N).
    For weighted aggregation proportional to local dataset size
    (the original FedAvg formulation), replace the uniform average with:
        w_global = Σ_i (n_i / n_total) * w_i
    """
    # Start with a deep copy of the first agent's weights
    avg_state_dict = copy.deepcopy(agent_list[0].model.state_dict())

    # Accumulate weights from all other agents
    for key in avg_state_dict:
        for i in range(1, len(agent_list)):
            avg_state_dict[key] += agent_list[i].model.state_dict()[key]
        # Divide by N to get the true mean
        avg_state_dict[key] = avg_state_dict[key] / len(agent_list)

    # Broadcast the global average to every agent
    for agent in agent_list:
        agent.model.load_state_dict(avg_state_dict)


def Hamiltonian_cycle_algorithm(agent_list):
    """
    Hamiltonian Cycle Weight Rotation (Ring-Topology Model Passing).

    Rotates model weights around a virtual ring: each agent receives
    the weights of its predecessor (the previous agent in the list).
    Over successive epochs, weights travel around the full cycle,
    allowing each model to visit every node exactly once per full rotation.

    Mathematical Foundation
    -----------------------
    This is a **Hamiltonian cycle gossip** strategy, a structured variant
    of model-passing protocols studied in decentralized federated learning
    (Wang et al., 2022 ; DRDFL, 2024).

    A Hamiltonian cycle on N nodes is a cycle that visits every node
    exactly once. In this weight-sharing context:

        w_i(t+1) ← w_{(i-1) mod N}(t)

    This is NOT an averaging operation — it is a pure rotation.
    The key property: after exactly N rotation steps, every agent
    will have seen (held) every other agent's initial model.

    This strategy is communication-efficient: each agent sends and
    receives exactly one model per round (degree = 2 in the ring),
    minimizing bandwidth while ensuring full coverage over N rounds.

    Convergence
    -----------
    Unlike averaging-based consensus, pure rotation does not converge
    to the global average by itself. It is typically used as a building
    block for hybrid strategies (see `Hamiltonian_cycle_algorithm_hybride_consensus`),
    where rotation spreads diversity across agents before a final
    averaging consensus step collapses them to a shared optimum.

    Reference
    ---------
    - Wang, Z. et al. (2022). "Efficient ring-topology decentralized
      federated learning with deep generative models for medical data."
      Electronics, 11(10), 1548.
    - DRDFL (2024). "Divide-and-Conquer Collaboration for Ring-Topology
      Decentralized Federated Learning." OpenReview.

    Parameters
    ----------
    agent_list : list[Agent]
        Ordered list of agents forming the ring. The ring is defined by
        list order: agent[0] → agent[1] → ... → agent[N-1] → agent[0].

    Returns
    -------
    None
        Each agent's model weights are replaced in-place by its
        predecessor's weights.

    Warning
    -------
    This function takes a full snapshot of ALL weights before any
    update, ensuring the rotation is truly simultaneous. Without this,
    agent[1] would receive already-updated weights from agent[0].
    """
    # Snapshot all weights before any modification
    state_dict_list = [
        copy.deepcopy(agent.model.state_dict())
        for agent in agent_list
    ]

    # Rotate: agent i receives the weights of agent (i-1) mod N
    for e, agent in enumerate(agent_list):
        agent.model.load_state_dict(
            state_dict_list[(e + 1) % len(agent_list)]
        )


count_epoches = 0


def Hamiltonian_cycle_algorithm_hybride_consensus(agent_list, K, epoch, num_epochs):
    """
    Hybrid Strategy: Hamiltonian Rotation + Final Average Consensus.

    Combines two complementary mechanisms into a two-phase training strategy:

    - **Phase 1 (epochs 0 to N-3)**: Hamiltonian rotation — agents pass
      their weights along a ring, promoting diversity and exploration.
    - **Phase 2 (last 2 epochs)**: Average consensus on the Hamiltonian
      ring graph — agents converge to a shared global model via iterative
      averaging.

    Motivation & Mathematical Foundation
    -------------------------------------
    This hybrid approach is inspired by the observation that pure averaging
    too early in training can collapse agent diversity before local models
    have had a chance to specialize, while pure rotation never converges
    to a global optimum (Beltrán et al., 2023 ; DRDFL, 2024).

    The two-phase strategy can be viewed as:

    Phase 1 — Exploration (rotation):
        w_i(t+1) ← w_{(i-1) mod N}(t)

        Models travel around the ring, each agent acting as a "relay"
        that trains a visiting model on its local data. This implicitly
        trains each model on a diverse sequence of local datasets.

    Phase 2 — Exploitation (consensus):
        R(k+1) = P · R(k),   k = 0, …, K-1

        where P is the doubly stochastic mixing matrix of the Hamiltonian
        cycle graph. This drives all agents toward the global average of
        the models accumulated during Phase 1.

    The Hamiltonian cycle graph has a known spectral gap:
        λ₂(P) = cos(2π/N)

    which is close to 1 for large N (slow convergence) — hence why K
    iterations of consensus are needed at the end.

    Reference
    ---------
    - Beltrán, E.T.M. et al. (2023). "Decentralized Federated Learning:
      Fundamentals, State of the Art, Frameworks, Trends, and Challenges."
      IEEE Communications Surveys & Tutorials.
    - Wang, Z. et al. (2022). "Efficient ring-topology decentralized
      federated learning." Electronics, 11(10), 1548.
    - DRDFL (2024). "Divide-and-Conquer Collaboration for Ring-Topology
      Decentralized Federated Learning." OpenReview.

    Parameters
    ----------
    agent_list : list[Agent]
        List of all agents participating in federated training.
    K : int
        Number of consensus iterations to run during Phase 2.
        Higher K → closer to global average, more communication.
    epoch : int
        Current training epoch (0-indexed).
    num_epochs : int
        Total number of training epochs.

    Returns
    -------
    r : dict[int, dict[str, torch.Tensor]] or None
        - During Phase 1: returns None (rotation only, in-place update).
        - During Phase 2: returns the final averaged state dicts from
          `Average_consensus_algorithm`, keyed by agent id.

    Phase Transition Logic
    ----------------------
    - epoch < num_epochs - 2  →  Phase 1: Hamiltonian rotation
    - epoch >= num_epochs - 2 →  Phase 2: Average consensus on ring

    Example
    -------
    For num_epochs = 10:
        Epochs 0–7  : Phase 1 (rotation)
        Epochs 8–9  : Phase 2 (consensus, K iterations each)
    """
    n = len(agent_list)
    # Build the Hamiltonian cycle graph for Phase 2
    hamil_graph = NetworkTopology.cycle_graph_amiltionen(n)

    if epoch < num_epochs - 2:
        # ── Phase 1 : Hamiltonian rotation ────────────────────────────────
        print("    [Hybrid] Phase 1 — Hamiltonian rotation …")
        snapshots = [
            copy.deepcopy(agent.model.state_dict())
            for agent in agent_list
        ]
        # Each agent i receives the model from agent (i-1) mod N
        for i, agent in enumerate(agent_list):
            predecessor = (i - 1) % n
            agent.model.load_state_dict(snapshots[predecessor])

    else:
        # ── Phase 2 : Average consensus on the ring ────────────────────────
        print("    [Hybrid] Phase 2 — Average consensus on Hamiltonian ring …")
        r = Average_consensus_algorithm(agent_list, hamil_graph, K=K)
        return r