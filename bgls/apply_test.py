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

"""Tests for applying operations to states."""

import pytest
import numpy as np

import cirq

import bgls


def test_act_on_stabilizer_non_T():
    """Assertion thrown if used on a non-T non-Rz non-Clifford gate"""
    qubits = cirq.LineQubit.range(3)
    state = cirq.StateVectorSimulationState(qubits=qubits, initial_state=0)
    # fails for non zpowgate
    with pytest.raises(ValueError):
        gate = cirq.XPowGate(exponent=0.1)
        bgls.apply.act_on_near_clifford(gate.on(qubits[0]), state)


@pytest.mark.parametrize("rads", [0.0, 0.5, 1.0, 1.5, np.pi + 0.1])
def test_generic_rz_gates(rads: float):
    """act_on_near_clifford supports other rz rotations besides T (pi/4)"""
    qubits = cirq.LineQubit.range(3)
    state1 = cirq.StateVectorSimulationState(qubits=qubits, initial_state=0)
    state2 = cirq.StateVectorSimulationState(qubits=qubits, initial_state=0)
    state2S = state2.copy()
    cirq.act_on(cirq.S.on(qubits[0]), state2S)

    op = cirq.Rz(rads=rads)
    bgls.apply.act_on_near_clifford(op.on(qubits[0]), state1)
    assert (state1.target_tensor == state2.target_tensor).all() or (
        state1.target_tensor == state2S.target_tensor
    ).all()


@pytest.mark.parametrize("channel", (cirq.depolarize, cirq.amplitude_damp))
def test_act_on_noisy_state_vector(channel):
    """Tests correctness of simulating noisy circuits with
    common single-qubit channels.
    """
    qubits = cirq.LineQubit.range(2)

    circuit = cirq.testing.random_circuit(qubits, n_moments=3, op_density=1)
    circuit = circuit.with_noise(channel(0.01))

    sim = bgls.Simulator(
        initial_state=cirq.StateVectorSimulationState(
            qubits=qubits, initial_state=0
        ),
        apply_op=bgls.apply.act_on_noisy_state_vector,
        compute_probability=bgls.born.compute_probability_state_vector,
    )
    sim_cirq = cirq.Simulator()

    # Test expectation of observables match Cirq.Simulator.
    observables = [cirq.X.on(qubits[0]), cirq.Z.on(qubits[1])]

    values = sim.sample_expectation_values(
        circuit,
        observables=observables,
        num_samples=2048,
    )
    values_cirq = sim_cirq.sample_expectation_values(
        circuit,
        observables=observables,
        num_samples=2048,
    )
    assert np.allclose(values, values_cirq, atol=1e-1)
