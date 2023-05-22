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

"""Tests for the BGLS Simulator utilities."""

import pytest

import cirq

import bgls


@pytest.mark.parametrize("nqubits", range(3, 5 + 1))
def test_improved_random_circuit(nqubits: int):
    """Our random circuit function matches cirqs' output."""
    qubits = cirq.LineQubit.range(nqubits)
    domain = {cirq.H, cirq.CNOT, cirq.S, cirq.T}
    domain_cirq = {cirq.H: 1, cirq.CNOT: 2, cirq.S: 1, cirq.T: 1}
    clifford_circuit = bgls.utils.generate_random_circuit(
        qubits,
        n_moments=100,
        op_density=0.5,
        gate_domain=domain,
        random_state=1,
    )
    clifford_circuit2 = cirq.testing.random_circuit(
        qubits,
        n_moments=100,
        op_density=0.5,
        gate_domain=domain_cirq,
        random_state=1,
    )
    assert clifford_circuit2 == clifford_circuit


def test_act_on_stabilizer_non_T():
    """Assertion thrown if used on a non-T non-Clifford gate"""
    qubits = cirq.LineQubit.range(3)
    state = cirq.StateVectorSimulationState(qubits=qubits, initial_state=0)
    # fails for non zpowgate
    with pytest.raises(AssertionError):
        gate = cirq.XPowGate(exponent=0.1)
        bgls.utils.act_on_near_clifford(gate.on(qubits[0]), state)
    # and for zpowgate of the wrong exponent
    with pytest.raises(AssertionError):
        gate = cirq.ZPowGate(exponent=0.2)
        bgls.utils.act_on_near_clifford(gate.on(qubits[1]), state)
