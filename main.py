# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import torch
import copy
import numpy as np


class DecentralizedFederatedLearning:
    """
    Implementation of Algorithm 1:
    Decentralized Federated Learning based on Average Consensus
    """

    def __init__(self, agents, W, K, L):
        """
        Args:
            agents: List of N agents
            W: N×N mixing matrix (consensus weights)
            K: Number of consensus iterations per round
            L: Number of rounds
        """
        self.agents = agents
        self.N = len(agents)
        self.W = W  # Mixing matrix
        self.K = K  # Consensus iterations per round
        self.L = L  # Total rounds

        # Storage for history
        self.history = {
            'pn': [],  # After consensus
            'qn': [],  # After local training
            'divergence': []
        }

    def initialize(self):
        """Line 1: Initialize all agents"""
        print("🔧 Initializing all agents...")
        for n, agent in enumerate(self.agents):
            # pn(0) = initial parameters
            agent.pn = copy.deepcopy(agent.model.state_dict())
            print(f"   Agent {n}: initialized")

    def local_training(self, round_num):
        """Lines 3-5: Local training phase"""
        print(f"\n📚 Round {round_num}: LOCAL TRAINING")
        print(f"   {'=' * 60}")

        for n, agent in enumerate(self.agents):
            # Line 4: Train agent n using its local data
            # Input: pn(ℓ-1) from previous round
            # Output: qn(ℓ) after local training

            print(f"   Agent {n}:")
            print(f"      Input:  pn({round_num - 1})")

            # Load previous round's consensed parameters
            agent.model.load_state_dict(agent.pn)

            # Perform local training (e.g., 1 epoch)
            train_loss, train_acc = agent.train_one_epoch()

            # Store qn(ℓ) = trained parameters
            agent.qn = copy.deepcopy(agent.model.state_dict())

            print(f"      Local training: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"      Output: qn({round_num})")

    def consensus_algorithm(self, round_num):
        """Lines 6-16: Consensus update phase (synchronous)"""
        print(f"\n🔗 Round {round_num}: CONSENSUS ({self.K} iterations)")
        print(f"   {'=' * 60}")

        # Line 7: Initialize consensus variable
        # r_n^(ℓ)(0) ← qn(ℓ)
        print(f"\n   Initialization (k=0):")
        r_current = []
        for n, agent in enumerate(self.agents):
            r_current.append(copy.deepcopy(agent.qn))
            print(f"      Agent {n}: r_n^({round_num})(0) ← qn({round_num})")

        # Lines 9-13: Consensus iterations
        print(f"\n   Consensus iterations (k=1 to {self.K}):")

        for k in range(1, self.K + 1):
            print(f"\n      Iteration k={k}:")
            r_next = []

            # Line 10-11: Each agent updates using weighted average
            for n, agent in enumerate(self.agents):
                # ✅ SYNCHRONOUS UPDATE
                # r_n^(ℓ)(k) = w_nn·r_n^(ℓ)(k-1) + ∑[m∈N_n] w_nm·r_m^(ℓ)(k-1)

                # Start with self weight
                weights_to_avg = [
                    {key: self.W[n, n] * r_current[n][key]
                     for key in r_current[n]}
                ]

                # Add neighbors' weighted parameters
                for m in self.agents[n].neighbors:
                    weights_to_avg.append(
                        {key: self.W[n, m] * r_current[m][key]
                         for key in r_current[m]}
                    )

                # Average
                r_next_n = self._average_params(weights_to_avg)
                r_next.append(r_next_n)

                print(f"         Agent {n}: r_n^({round_num})({k}) ← " +
                      f"averaged from self + {len(self.agents[n].neighbors)} neighbors")

            # ✅ ALL agents update SIMULTANEOUSLY
            r_current = r_next

            # Measure divergence
            divergence = self._compute_consensus_divergence(r_current)
            print(f"         Global divergence: {divergence:.8f}")

        # Line 15: Finalize consensus
        # pn(ℓ) ← r_n^(ℓ)(K)
        print(f"\n   Finalization:")
        for n, agent in enumerate(self.agents):
            agent.pn = r_current[n]
            print(f"      Agent {n}: pn({round_num}) ← r_n^({round_num})({self.K})")

    def train(self):
        """Main training loop: Line 2 (for ℓ = 1 to L do)"""

        print(f"\n{'#' * 70}")
        print(f"# DECENTRALIZED FEDERATED LEARNING")
        print(f"# Parameters: N={self.N}, K={self.K}, L={self.L}")
        print(f"{'#' * 70}")

        # Line 1: Initialize
        self.initialize()

        # Line 2: For each round ℓ
        for ell in range(1, self.L + 1):
            print(f"\n{'#' * 70}")
            print(f"# ROUND ℓ = {ell}/{self.L}")
            print(f"{'#' * 70}")

            # Local training phase
            self.local_training(ell)

            # Consensus phase
            self.consensus_algorithm(ell)

            # Validation & metrics
            self._validate_and_log(ell)

            print(f"\n✅ Round {ell} complete")

        # Line 18: Return final parameters
        print(f"\n{'#' * 70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'#' * 70}")
        return self._get_final_parameters()

    def _average_params(self, weighted_params_list):
        """Average a list of weighted parameter dictionaries"""
        averaged = {}
        for key in weighted_params_list[0]:
            averaged[key] = weighted_params_list[0][key].clone()
            for i in range(1, len(weighted_params_list)):
                averaged[key] += weighted_params_list[i][key]
        return averaged

    def _compute_consensus_divergence(self, params_list):
        """Compute ||θ_i - θ_avg||^2"""
        # Average all parameters
        avg_params = {}
        for key in params_list[0]:
            avg_params[key] = sum(p[key] for p in params_list) / len(params_list)

        # Compute divergence for each agent
        total_divergence = 0
        for params in params_list:
            for key in params:
                diff = params[key] - avg_params[key]
                total_divergence += torch.norm(diff).item() ** 2

        return total_divergence / len(params_list)

    def _validate_and_log(self, round_num):
        """Validation and logging"""
        print(f"\n📊 Round {round_num}: VALIDATION & METRICS")
        print(f"   {'=' * 60}")

        for n, agent in enumerate(self.agents):
            agent.model.load_state_dict(agent.pn)
            val_loss, val_acc = agent.validate_one_epoch()
            print(f"   Agent {n}: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            self.history['pn'].append(agent.pn)

        # Global consensus divergence
        divergence = self._compute_consensus_divergence(
            [agent.pn for agent in self.agents]
        )
        self.history['divergence'].append(divergence)
        print(f"   Global Consensus Divergence: {divergence:.8f}")

    def _get_final_parameters(self):
        """Line 18: Return p1(L), p2(L), ..., pN(L)"""
        return [agent.pn for agent in self.agents]

    def create_mixing_matrix(self, topology="ring"):
        """
        Create mixing matrix W for consensus

        Properties:
        - Each row sums to 1 (row-stochastic)
        - w_nn > 0, w_nm ≥ 0
        - w_nm > 0 iff m ∈ N_n or n = m
        """
        W = np.zeros((self.N, self.N))

        if topology == "ring":
            for n in range(self.N):
                neighbors = [(n - 1) % self.N, (n + 1) % self.N]
                W[n, n] = 0.5
                W[n, neighbors[0]] = 0.25
                W[n, neighbors[1]] = 0.25

        elif topology == "fully_connected":
            for n in range(self.N):
                W[n, :] = 1 / self.N

        # Verify row-stochastic property
        assert np.allclose(W.sum(axis=1), 1), "W must be row-stochastic"

        return W

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("hello world")