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

from typing import Callable

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
