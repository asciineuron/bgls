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

"""Tests for computing probabilities of bitstrings in given states."""

import numpy as np

import cirq

import bgls


def test_compute_probability_statevector():
    state = cirq.StateVectorSimulationState(qubits=cirq.LineQubit.range(2), initial_state=0b00)

    assert np.isclose(bgls.born.cirq_state_vector_bitstring_probability(state, "00"), 1.0)
    assert np.isclose(bgls.born.cirq_state_vector_bitstring_probability(state, "01"), 0.0)
    assert np.isclose(bgls.born.cirq_state_vector_bitstring_probability(state, "10"), 0.0)
    assert np.isclose(bgls.born.cirq_state_vector_bitstring_probability(state, "11"), 0.0)


def test_compute_probability_density_matrix():
    state = cirq.DensityMatrixSimulationState(qubits=cirq.LineQubit.range(2), initial_state=0b01)

    assert np.isclose(bgls.born.cirq_density_matrix_bitstring_probability(state, "00"), 0.0)
    assert np.isclose(bgls.born.cirq_density_matrix_bitstring_probability(state, "01"), 1.0)
    assert np.isclose(bgls.born.cirq_density_matrix_bitstring_probability(state, "10"), 0.0)
    assert np.isclose(bgls.born.cirq_density_matrix_bitstring_probability(state, "11"), 0.0)


def test_compute_probability_stabilizer_state():
    state = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(2), initial_state=0b10,
        )

    assert np.isclose(bgls.born.cirq_stabilizer_ch_bitstring_probability(state, "00"), 0.0)
    assert np.isclose(bgls.born.cirq_stabilizer_ch_bitstring_probability(state, "01"), 0.0)
    assert np.isclose(bgls.born.cirq_stabilizer_ch_bitstring_probability(state, "10"), 1.0)
    assert np.isclose(bgls.born.cirq_stabilizer_ch_bitstring_probability(state, "11"), 0.0)
