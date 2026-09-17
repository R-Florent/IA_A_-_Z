import copy

from sympy.codegen.ast import none
from agents.Classe_agent import Agent
from topologies.NetworkTopology import NetworkTopology
import copy
import torch
from hyperparametres import NUM_EPOCHES


def Sequential_AC(agent_list, graph,comm_cost):
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
    Because only `agent` is updated (not `neighbor`), the mixing m          atrix
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

    total_messages = sum(dict(graph.degree()).values())  # = nb_arêtes × 2
    comm_cost.set_epoch_messages(total_messages)


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


def Average_consensus_algorithm(agent_list, graph, comm_cost=none):
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
    if comm_cost is not None:
        total_messages = sum(dict(graph.degree()).values())  # = nb_arêtes × 2
        comm_cost.set_epoch_messages(total_messages)

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


def Average_consensus_algorithm_K(agent_list, graph, K=5, comm_cost=none):
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
    if comm_cost is not None:
        messages_per_iteration = sum(dict(graph.degree()).values())
        comm_cost.set_epoch_messages(messages_per_iteration * K)

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


def avg_models_fully_connected_graph_algorithm(agent_list,comm_cost=none):
    """
    Centralize
    d Federated Averaging (FedAvg — Global Aggregation Step).

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
    n = len(agent_list)

    if comm_cost is not None:
        comm_cost.set_epoch_messages(n * 2)  # upload + download par agent

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


def Hamiltonian_cycle_algorithm(agent_list,comm_cost):
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
    n = len(agent_list)
    if comm_cost is not None:
        comm_cost.set_epoch_messages(n * 1)  # 1 message reçu par agent

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


def Hamiltonian_cycle_algorithm_hybride_consensus(agent_list, K, epoch, num_epochs,comm_cost=None):
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
        if comm_cost is not None:
            comm_cost.set_epoch_messages(n)
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

        r = Average_consensus_algorithm_K(
            agent_list,
            hamil_graph,
            K=K,
            comm_cost=comm_cost
        )
        return r

def metropolis_weights(agent_list,graph, comm_cost=none):
    """
    Compute the Metropolis-Hastings Optimal Mixing Matrix.

    Mathematical Foundation
    -----------------------
    The speed of ANY consensus algorithm depends on λ₂(W),
    the second largest eigenvalue of the mixing matrix W.
    The smaller λ₂, the faster the convergence.

    The Metropolis-Hastings rule gives a simple CLOSED-FORM doubly
    stochastic matrix that near-optimally minimizes λ₂:

        W_ij = 1 / (1 + max(deg(i), deg(j)))   if (i,j) ∈ E
        W_ii = 1 - Σ_{j ∈ N_i} W_ij            (self-weight)
        W_ij = 0                                 otherwise

    Analogy:
        If two neighbors have different numbers of friends, the one
        with MORE friends gives LESS weight to each connection.
        This balances the influence and creates a doubly stochastic W
        without any global coordination — each agent only needs to
        know its own degree and its neighbor's degree.

    Reference
    ---------
    Xiao, L., Boyd, S. (2004). "Fast linear iterations for distributed
    averaging." Systems & Control Letters.

    Parameters
    ----------
    graph : networkx.Graph
    n     : int, number of agents

    Returns
    -------
    W : dict[int, dict[int, float]]
        Mixing matrix as nested dict. W[i][j] = weight agent i gives
        to agent j's model.
    """
    n = len(agent_list)

    if comm_cost is not None:
        total_messages = sum(dict(graph.degree()).values())  # = nb_arêtes × 2
        comm_cost.set_epoch_messages(total_messages)


    W = {i: {j: 0.0 for j in range(n)} for i in range(n)}

    for i in range(n):
        neighbors = list(graph.neighbors(i))
        for j in neighbors:
            deg_i = graph.degree(i)
            deg_j = graph.degree(j)
            W[i][j] = 1.0 / (1 + max(deg_i, deg_j))
        # Self-weight: 1 - sum of outgoing weights
        W[i][i] = 1.0 - sum(W[i][j] for j in neighbors)

    return W

def push_sum_consensus(agent_list, graph, K=10,comm_cost=none):
    """
    Push-Sum Consensus Algorithm for Directed Graphs.

    Mathematical Foundation
    -----------------------
    Standard averaging (DSGD) requires a SYMMETRIC, doubly stochastic
    matrix W. This breaks on directed graphs (e.g., agent A → B but
    not B → A).

    Push-Sum solves this by tracking two quantities per agent:
        - x_i(t) : weighted sum of model weights
        - w_i(t) : scalar weight (sum of mixing coefficients received)

    The ratio x_i(t) / w_i(t) converges to the global average:

        x_i(t+1) = Σ_{j: j→i}  (x_j(t) / out_degree(j))
        w_i(t+1) = Σ_{j: j→i}  (w_j(t) / out_degree(j))

        estimate_i(t) = x_i(t) / w_i(t)  →  (1/N) Σ_i x_i(0)

    Key property: works even if W is only COLUMN-stochastic
    (each agent distributes its full mass to neighbors).

    Reference
    ---------
    Kempe, D., Dobra, A., Gehrke, J. (2003). FOCS.
    Nedic, A., Olshevsky, A. (2015). IEEE Trans. Automatic Control.

    Parameters
    ----------
    agent_list : list[Agent]
    graph      : networkx.DiGraph  ← directed graph!
    K          : int, number of iterations

    Returns
    -------
    None — agents updated in-place with converged estimates.
    """
    n = len(agent_list)

    if comm_cost is not None:
        total_messages = sum(dict(graph.degree()).values())  # = nb_arêtes × 2
        comm_cost.set_epoch_messages(total_messages)

    # Initialize: x_i = model weights, w_i = 1.0 (scalar per agent)
    x = {
        agent.id: copy.deepcopy(agent.model.state_dict())
        for agent in agent_list
    }
    w = {agent.id: 1.0 for agent in agent_list}

    for k in range(K):
        x_new = {agent.id: None for agent in agent_list}
        w_new = {agent.id: 0.0 for agent in agent_list}

        for agent in agent_list:
            # Each agent splits its mass equally among out-neighbors + self
            out_neighbors = list(graph.neighbors(agent.id))
            recipients = [agent.id] + out_neighbors
            share = 1.0 / len(recipients)

            for recipient_id in recipients:
                # Accumulate x contribution
                contrib = {
                    key: x[agent.id][key] * share
                    for key in x[agent.id]
                }
                if x_new[recipient_id] is None:
                    x_new[recipient_id] = contrib
                else:
                    for key in contrib:
                        x_new[recipient_id][key] += contrib[key]

                # Accumulate w contribution
                w_new[recipient_id] += w[agent.id] * share

        x = x_new
        w = w_new

        # Compute estimates x_i / w_i and apply to models
        for agent in agent_list:
            estimate = {
                key: x[agent.id][key] / w[agent.id]
                for key in x[agent.id]
            }
            agent.model.load_state_dict(estimate)

        print(f"    [Push-Sum] iter {k+1}/{K} — "
              f"w_sum = {sum(w.values()):.4f} (should stay ≈ {float(n):.1f})")

    return x, w


def gradient_tracking_consensus(agent_list, graph, K=10, lr=0.01,comm_cost=None):
    """
    Gradient Tracking Consensus (DIGing / NEXT Algorithm).

    THE KEY PROBLEM THIS SOLVES
    ---------------------------
    Standard DSGD / average consensus only averages MODEL WEIGHTS.
    With non-IID data (each agent has different data distribution),
    the local gradients point in different directions.

    After averaging weights, each agent computes a gradient biased
    toward its own local data → the global model drifts away from
    the TRUE global optimum. This is called "gradient bias" or
    "client drift".

    Mathematical Foundation
    -----------------------
    Gradient Tracking maintains an auxiliary variable y_i that
    TRACKS the average gradient across all agents:

        Model update:
            x_i(t+1) = Σ_j W_ij · x_j(t) - lr · y_i(t)

        Gradient tracker update:
            y_i(t+1) = Σ_j W_ij · y_j(t)
                       + ∇f_i(x_i(t+1)) - ∇f_i(x_i(t))

    Intuition:
        y_i(t) is a "gradient memory" — it accumulates the difference
        between the new and old local gradients. Over iterations,
        y_i converges to the AVERAGE gradient (1/N) Σ_i ∇f_i(x).

        This makes each agent effectively descend along the GLOBAL
        gradient, not just its local one → exact convergence even
        with heterogeneous data.

    Key result (Nedic et al., 2017):
        With gradient tracking, the algorithm converges to the EXACT
        global optimum (not just a biased neighborhood of it).

    This is the main advantage over plain DSGD, which only finds
    an approximation when data is heterogeneous.

    Reference
    ---------
    - Nedic, A., Olshevsky, A., Shi, W. (2017). "Achieving geometric
      convergence for distributed optimization over time-varying
      graphs." SIAM J. Optimization.
    - Lorenzo, P.D., Scutari, G. (2016). "NEXT: In-network nonconvex
      optimization." IEEE Trans. Signal & Info. Processing over Networks.
    - Koloskova, A. et al. (2021). "An improved analysis of gradient
      tracking for decentralized ML." NeurIPS 2021.

    Parameters
    ----------
    agent_list     : list[Agent]
    graph          : networkx.Graph (undirected, connected)
    local_grad_fn  : callable(agent) → dict[str, Tensor]
                     Function that computes the local gradient for one
                     agent and returns it as a state-dict-like dict.
                     Example:
                         def local_grad_fn(agent):
                             loss = criterion(agent.model(X), y)
                             loss.backward()
                             return {n: p.grad.clone()
                                     for n, p in agent.model.named_parameters()}
    K              : int, number of iterations
    lr             : float, learning rate

    Returns
    -------
    None — agents updated in-place.
    """

    n = len(agent_list)

    if comm_cost is not None:
        comm_cost.set_epoch_messages(n)

    # Build mixing matrix W (Metropolis-Hastings weights — see algo 4)
    W = metropolis_weights(graph, n,metropolis_weights)

    # Initialize gradient trackers y_i = ∇f_i(x_i(0))
    y = {
        agent.id: local_grad_fn(agent)
        for agent in agent_list
    }

    for k in range(K):

        # ── Step 1 : gradient step using tracked gradient ──────────────
        old_grads = {
            agent.id: local_grad_fn(agent)
            for agent in agent_list
        }

        # Model update: x_i ← Σ_j W_ij x_j - lr * y_i
        x_new = {}
        for agent in agent_list:
            # Weighted sum of neighbor models
            mixed = {
                key: torch.zeros_like(agent.model.state_dict()[key])
                for key in agent.model.state_dict()
            }
            for other in agent_list:
                wij = W[agent.id][other.id]
                if wij > 0:
                    for key in mixed:
                        mixed[key] += wij * other.model.state_dict()[key]

            # Subtract gradient tracker contribution
            for key in mixed:
                if key in y[agent.id]:
                    mixed[key] -= lr * y[agent.id][key]

            x_new[agent.id] = mixed

        # Apply new model weights
        for agent in agent_list:
            agent.model.load_state_dict(x_new[agent.id])

        # ── Step 2 : update gradient trackers ─────────────────────────
        new_grads = {
            agent.id: local_grad_fn(agent)
            for agent in agent_list
        }

        y_new = {}
        for agent in agent_list:
            # Weighted sum of neighbor trackers
            y_mixed = {
                key: torch.zeros_like(y[agent.id][key])
                for key in y[agent.id]
            }
            for other in agent_list:
                wij = W[agent.id][other.id]
                if wij > 0:
                    for key in y_mixed:
                        y_mixed[key] += wij * y[other.id][key]

            # Add gradient correction: ∇f_i(new) - ∇f_i(old)
            for key in y_mixed:
                y_mixed[key] += new_grads[agent.id][key] - old_grads[agent.id][key]

            y_new[agent.id] = y_mixed

        y = y_new

        print(f"    [GradTrack] iteration {k+1}/{K} complete")



def exact_diffusion_consensus(agent_list, graph, K=10, comm_cost=none ):
    """
    Exact Diffusion (D² Algorithm) — Bias-Free Weight Consensus.

    Mathematical Foundation
    -----------------------
    Exact Diffusion corrects the well-known bias of standard diffusion/
    consensus by maintaining a CORRECTION TERM φ_i that accumulates
    the difference between successive local states.

    The update rule (Yuan et al., 2018) is:

        ψ_i(t+1) = x_i(t) + φ_i(t)          ← pre-combination step
        x_i(t+1) = Σ_j W_ij · ψ_j(t+1)      ← combination step
        φ_i(t+1) = ψ_i(t+1) - x_i(t+1) + φ_i(t)  ← correction update

    where φ_i(0) = 0.

    Intuition (analogy):
        Think of each agent as a boat on water. Standard consensus is
        like everyone rowing toward the average position — but if the
        current (gradient bias) pushes you sideways, you drift.
        Exact Diffusion is like each boat also tracking HOW MUCH it
        has been pushed sideways and correcting for it every step.

    Key property:
        Under exact diffusion, the consensus error converges to ZERO
        (not just close to zero), even with heterogeneous data and
        fixed mixing matrices.

    Compared to Gradient Tracking:
        - Exact Diffusion uses only model weights (no gradient tracking)
        - Simpler to implement, lower memory overhead
        - Also achieves exact convergence
        - Better suited for pure weight-averaging scenarios

    Reference
    ---------
    - Yuan, K., Ying, B., Zhao, X., Sayed, A.H. (2018).
      "Exact diffusion for distributed optimization and learning."
      IEEE Transactions on Signal Processing.

    Parameters
    ----------
    agent_list : list[Agent]
    graph      : networkx.Graph
    K          : int, iterations

    Returns
    -------
    None — agents updated in-place.
    """
    n = len(agent_list)
    W = metropolis_weights(agent_list,graph,comm_cost=comm_cost)

    if comm_cost is not None:
        comm_cost.set_epoch_messages(n)

    # Initialize correction terms φ_i = 0
    phi = {
        agent.id: {
            key: torch.zeros_like(param)
            for key, param in agent.model.state_dict().items()
        }
        for agent in agent_list
    }

    for k in range(K):

        # ── Step 1 : pre-combination ψ_i = x_i + φ_i ──────────────────
        psi = {}
        for agent in agent_list:
            x = agent.model.state_dict()
            psi[agent.id] = {
                key: x[key] + phi[agent.id][key]
                for key in x
            }

        # ── Step 2 : combination x_i = Σ_j W_ij ψ_j ──────────────────
        x_new = {}
        for agent in agent_list:
            combined = {
                key: torch.zeros_like(psi[agent.id][key])
                for key in psi[agent.id]
            }
            for other in agent_list:
                wij = W[agent.id][other.id]
                if wij > 0:
                    for key in combined:
                        combined[key] += wij * psi[other.id][key]
            x_new[agent.id] = combined

        # ── Step 3 : correction φ_i = ψ_i - x_i_new + φ_i ────────────
        for agent in agent_list:
            for key in phi[agent.id]:
                phi[agent.id][key] = (
                    psi[agent.id][key]
                    - x_new[agent.id][key]
                    + phi[agent.id][key]
                )

        # Apply updated weights
        for agent in agent_list:
            agent.model.load_state_dict(x_new[agent.id])

        print(f"    [ExactDiff] iteration {k+1}/{K} complete")