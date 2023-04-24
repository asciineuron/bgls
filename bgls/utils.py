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

import numpy as np

import cirq

import cirq.contrib.quimb.mps_simulator


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
    density_matrix_state: cirq.sim.DensityMatrixSimulationState, bitstring: str
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


def cirq_mps_bitstring_probability(
    mps: cirq.contrib.quimb.MPSState, bitstring: str
) -> float:
    """
    Returns the probability of measuring the `bitstring` (|z⟩) in the
    'cirq.contrib.quimb.MPSState' mps.
    Args:
        mps: Matrix Product State as a 'cirq.contrib.quimb.MPSState'.
        bitstring: Bitstring |z⟩ as a binary string.
    """
    # TODO investigate runtime and if there is a better way to do this
    state_vec = mps.state_vector()
    return np.abs(state_vec[int(bitstring, 2)]) ** 2
