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

"""Tests for circuit optimization."""

import pytest

import cirq

import bgls


@pytest.mark.parametrize("op_density", (0.01, 0.1, 1.0))
def test_optimized_circuit_is_unitarily_equivalent(op_density):
    """Tests the optimized circuit has the same unitary as the original circuit."""
    circuit = cirq.testing.random_circuit(
        qubits=6, n_moments=20, op_density=op_density, random_state=1
    )
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        bgls.optimize_for_bgls(circuit),
        circuit,
        qubit_map={q: q for q in circuit.all_qubits()},
    )


def test_optimize_empty_circuit():
    """Tests edge case on call to optimize with an empty circuit."""
    circuit = cirq.Circuit()
    assert bgls.optimize_for_bgls(circuit) is circuit
