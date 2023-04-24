# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines helper functions for simulation."""

from typing import Union, Sequence, Optional, Set, List

import numpy as np

import cirq

import bgls


def cirq_state_vector_bitstring_probability(
    state_vector_state: cirq.sim.state_vector_simulation_state.StateVectorSimulationState,
    bitstring: str,
) -> float:
    """Returns the probability of measuring the `bitstring` (|z⟩) in the
    `cirq.StateVectorSimulationState` (|ψ⟩), i.e. |⟨z|ψ⟩|^2.

    Args:
        state_vector_state: State vector |ψ⟩ as a
            `cirq.StateVectorSimulationState`.
        bitstring: Bitstring |z⟩ as a binary string.
    """
    return (
        np.abs(
            cirq.to_valid_state_vector(state_vector_state.target_tensor)[
                int(bitstring, 2)
            ]
        )
        ** 2
    )


def cirq_density_matrix_bitstring_probability(
    density_matrix_state: cirq.sim.DensityMatrixSimulationState,
    bitstring: str,
) -> float:
    """Returns the probability of measuring the `bitstring` (|z⟩) in the
    `cirq.DensityMatrixSimulationState` (ρ), i.e. ⟨z|ρ|z⟩.

    Args:
        density_matrix_state: Density matrix ρ as a
            `cirq.DensityMatrixSimulationState`.
        bitstring: Bitstring |z⟩ as a binary string.
    """
    num_qubits = len(bitstring)
    index = int(bitstring, 2)
    density_matrix = cirq.to_valid_density_matrix(
        density_matrix_state.target_tensor, num_qubits
    )
    return np.abs(density_matrix[index, index])


def cirq_stabilizer_ch_bitstring_probability(
    stabilizer_ch_form_state: cirq.sim.StabilizerChFormSimulationState,
    bitstring: str,
) -> float:
    """Returns the probability of measuring the `bitstring` (|z⟩) in the
    `cirq.StabilizerChFormSimulationState` (U_C U_H|s⟩), i.e. |⟨z|ψ⟩|^2.

    Args:
        stabilizer_ch_form_state: Stabilizer state in CH form (arxiv:
        1808.00128) as a 'cirq.StabilizerChFormSimulationState'.
        bitstring: Bitstring |z⟩ as a binary string.

    """
    # the state is of type StabilizerStateChForm
    state = stabilizer_ch_form_state.state
    index = int(bitstring, 2)
    # this runs in O(n^2) for an n qubit state
    return np.abs(state.inner_product_of_state_and_x(index)) ** 2


def apply_near_clifford_gate(
    op: cirq.Operation,
    state: bgls.simulator.State,
    rng: np.random.RandomState = np.random.RandomState(),
) -> None:
    """
    Applies a Clifford+T gate to a state. If the gate is Clifford,
    apply normally, else choose one of the gates in the Clifford-expansion
    to apply.

    Parameters:
        op: operation to apply.
        state: simulator state to update.
        rng: rng for controlling gate-expansion decisions.
    """
    if cirq.has_stabilizer_effect(op):
        cirq.protocols.act_on(op, state)
    else:
        # assuming T gate
        # assert isinstance(op,
        #                   cirq.ops.common_gates.ZPowGate) and \
        #        op.gate.exponent == 0.25
        probs = np.power(
            np.abs(
                [
                    np.cos(np.pi / 8) - np.sin(np.pi / 8),
                    np.sqrt(2.0)
                    * np.exp(-(0 + 1j) * np.pi / 4)
                    * np.sin(np.pi / 8),
                ]
            ),
            2,
        )
        cand_gates = [cirq.I(op.qubits[0]), cirq.S(op.qubits[0])]
        chosen_gate = rng.choice(cand_gates, p=probs / sum(probs))
        cirq.protocols.act_on(chosen_gate, state)


def improved_random_circuit(
    qubits: Union[Sequence[cirq.ops.Qid], int],
    n_moments: int,
    op_density: float,
    gate_domain: Optional[Set[cirq.Gate]] = None,
    random_state: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
) -> cirq.circuits.Circuit:
    """Generates a random circuit.

    Args:
        qubits: If a sequence of qubits, then these are the qubits that
            the circuit should act on. Because the qubits on which an
            operation acts are chosen randomly, not all given qubits
            may be acted upon. If an int, then this number of qubits will
            be automatically generated, and the qubits will be
            `cirq.NamedQubits` with names given by the integers in
            `range(qubits)`.
        n_moments: The number of moments in the generated circuit.
        op_density: The probability that a gate is selected to operate on
            randomly selected qubits. Note that this is not the expected number
            of qubits that are acted on, since there are cases where the
            number of qubits that a gate acts on does not evenly divide the
            total number of qubits.
        gate_domain: The set of gates to choose from, specified as a set of
        gates. If not provided, the default gate
            domain is
            {X, Y, Z, H, S, T, CNOT, CZ, SWAP, ISWAP, CZPowGate()}. Only gates
            which act on a number of qubits less than len(qubits) (or qubits if
            provided as an int) are selected from the gate domain.
        random_state: Random state or random state seed.

    Raises:
        ValueError:
            * op_density is not in (0, 1].
            * gate_domain is empty.
            * qubits is an int less than 1 or an empty sequence.

    Returns:
        The randomly generated Circuit.
    """
    if not 0 < op_density <= 1:
        raise ValueError(f"op_density must be in (0, 1] but was {op_density}.")
    if gate_domain is None:
        gate_domain = {
            cirq.X,
            cirq.Y,
            cirq.Z,
            cirq.H,
            cirq.S,
            cirq.T,
            cirq.CNOT,
            cirq.CZ,
            cirq.SWAP,
            cirq.ISWAP,
            cirq.CZPowGate(),
        }
    if not gate_domain:
        raise ValueError("gate_domain must be non-empty.")

    if isinstance(qubits, int):
        qubits = tuple(cirq.ops.NamedQubit(str(i)) for i in range(qubits))
    n_qubits = len(qubits)
    if n_qubits < 1:
        raise ValueError("At least one qubit must be specified.")
    gate_domain = {g for g in gate_domain if g.num_qubits() <= n_qubits}
    if not gate_domain:
        raise ValueError(
            f"After removing gates that act on less than "
            f"{n_qubits} qubits, gate_domain had no gates."
        )
    max_arity = max([g.num_qubits() for g in gate_domain])

    prng = cirq.value.parse_random_state(random_state)

    moments: List[cirq.circuits.Moment] = []
    gate_list = sorted(gate_domain, key=repr)
    num_gates = len(gate_domain)
    for _ in range(n_moments):
        operations = []
        free_qubits = set(qubits)
        while len(free_qubits) >= max_arity:
            gate = gate_list[prng.randint(num_gates)]
            op_qubits = prng.choice(
                sorted(free_qubits),
                size=gate.num_qubits(),
                replace=False
                # type: ignore[arg-type]
            )
            free_qubits.difference_update(op_qubits)
            if prng.rand() <= op_density:
                operations.append(gate(*op_qubits))
        moments.append(cirq.circuits.Moment(operations))

    return cirq.circuits.Circuit(moments)
