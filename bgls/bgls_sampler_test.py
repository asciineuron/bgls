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

"""Tests for the BGLS Simulator."""

import pytest

import cirq

import bgls


@pytest.mark.parametrize("nqubits", range(3, 8 + 1))
def test_samples_correct_bitstrings_for_ghz_circuit(nqubits: int):
    """Tests correct measurement results for a GHZ circuit (should only return
    the all 0 or all 1 bitstring).

    Args:
        nqubits: Number of qubits in GHZ circuit.
    """
    qubits = cirq.LineQubit.range(nqubits)
    circuit = cirq.Circuit(
        cirq.H.on(qubits[0]),
        (cirq.CNOT.on(qubits[i], qubits[i + 1]) for i in range(nqubits - 1)),
        cirq.measure(qubits, key="z"),
    )

    sim = bgls.Simulator(
        initial_state=cirq.StateVectorSimulationState(
            qubits=qubits, initial_state=0
        ),
        apply_gate=cirq.protocols.act_on,
        compute_probability=bgls.state_vector_bitstring_probability,
    )
    results = sim.run(circuit, repetitions=100)
    measurements = set(results.histogram(key="z").keys())
    assert measurements.issubset({0, 2**nqubits - 1})


def test_results_same_when_seeded():
    """Tests simulator results are the same when provided the same seed."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H.on(q), cirq.measure(q, key="z"))

    sim_params = (
        cirq.StateVectorSimulationState(qubits=(q,), initial_state=0),
        cirq.protocols.act_on,
        bgls.state_vector_bitstring_probability,
    )
    sim1 = bgls.Simulator(*sim_params, seed=1)
    sim2 = bgls.Simulator(*sim_params, seed=1)

    result1 = sim1.run(circuit, repetitions=100)
    result2 = sim2.run(circuit, repetitions=100)

    assert result2 == result1


def test_intermediate_measurements():
    """Test simulation with/without intermediate measurements is the same."""
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2),
        cirq.measure([q0, q1, q2], key="result"),
    )
    ghz_intermediate = cirq.Circuit(
        cirq.H(q0),
        cirq.measure([q0, q2], key="result"),
        cirq.CNOT(q0, q1),
        cirq.measure([q0, q1, q2], key="result"),
        cirq.CNOT(q1, q2),
        cirq.measure([q0, q1, q2], key="result"),
    )

    sim = bgls.Simulator(
        cirq.StateVectorSimulationState(
            qubits=(q0, q1, q2), initial_state=0
        ),
        cirq.protocols.act_on,
        bgls.state_vector_bitstring_probability,
        seed=1,
    )
    result = sim.run(ghz, repetitions=100)

    sim = bgls.Simulator(
        cirq.StateVectorSimulationState(
            qubits=(q0, q1, q2), initial_state=0
        ),
        cirq.protocols.act_on,
        bgls.state_vector_bitstring_probability,
        seed=1,
    )
    result_with_intermediate_measurements = sim.run(
        ghz_intermediate, repetitions=100
    )
    assert result_with_intermediate_measurements == result


# def test_multiple_measurements():
#     # Distributing final measurement across multiple gates has same behavior
#     # as a single measurement of all qubits
#     q0, q1, q2 = cirq.LineQubit.range(3)
#     ghz = cirq.Circuit(
#         cirq.H(q0),
#         cirq.CNOT(q0, q1),
#         cirq.CNOT(q0, q2),
#         cirq.measure([q0, q1, q2], key="result"),
#     )
#     ghz_multiple = cirq.Circuit(
#         cirq.H(q0),
#         cirq.CNOT(q0, q1),
#         cirq.CNOT(q0, q2),
#         cirq.measure([q0, q2], key="result1st"),
#         cirq.measure([q1], key="result2nd"),
#     )
#     init_state = cirq.StateVectorSimulationState(
#         qubits=(q0, q1, q2), initial_state=0
#     )
#     bgls_simulator = bgls.Simulator(
#         init_state,
#         bgls.state_vector_bitstring_probability,
#         cirq.protocols.act_on,
#     )
#     bgls_result = bgls_simulator.sample(ghz, repetitions=100, seed=12)
#     bgls_result_multiple = bgls_simulator.sample(
#         ghz_multiple, repetitions=100, seed=12
#     )
#     # converting to histogram for checking recovers original values since
#     # spread across multiple measurement keys
#     np.testing.assert_array_equal(
#         cirq.vis.get_state_histogram(bgls_result),
#         cirq.vis.get_state_histogram(bgls_result_multiple),
#     )
#
#
# def test_no_measurements():
#     # Sampling a circuit without final measurement raises a ValueError
#     q0, q1, q2 = cirq.LineQubit.range(3)
#     ghz = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q0, q2))
#     init_state = cirq.StateVectorSimulationState(
#         qubits=(q0, q1, q2), initial_state=0
#     )
#     bgls_simulator = bgls.Simulator(
#         init_state,
#         bgls.state_vector_bitstring_probability,
#         cirq.protocols.act_on,
#     )
#     with pytest.raises(ValueError) as error_info:
#         bgls_result = bgls_simulator.sample(
#             ghz,
#             repetitions=100,
#         )
#
#
# def test_partial_measurements():
#     # Measuring only some final qubits yields the same distribution,
#     # but with bitstring int rep matching the subset of measured bits
#     q0, q1, q2 = cirq.LineQubit.range(3)
#     ghz = cirq.Circuit(
#         cirq.H(q0),
#         cirq.CNOT(q0, q1),
#         cirq.CNOT(q0, q2),
#         cirq.measure([q0, q2], key="result"),
#     )
#     init_state = cirq.StateVectorSimulationState(
#         qubits=(q0, q1, q2), initial_state=0
#     )
#     bgls_simulator = bgls.Simulator(
#         init_state,
#         bgls.state_vector_bitstring_probability,
#         cirq.protocols.act_on,
#     )
#     bgls_result = bgls_simulator.sample(
#         ghz,
#         repetitions=100,
#     )
#     # here because ghz, still only get low and high-end values with equal
#     # prob, but now 0 and 3 (ie 00 11) rather than 0 and 7 (000 111)
#     # can assert these are the only measured values
#     assert set(bgls_result.histogram(key="result")).issubset({0, 3})
#
#
# def test_density_state():
#     # Density matrix simulation yields the same result as state vector
#     q0, q1, q2 = cirq.LineQubit.range(3)
#     ghz = cirq.Circuit(
#         cirq.H(q0),
#         cirq.CNOT(q0, q1),
#         cirq.CNOT(q0, q2),
#         cirq.measure([q0, q1, q2], key="result"),
#     )
#     statevector_state = cirq.StateVectorSimulationState(
#         qubits=(q0, q1, q2), initial_state=0
#     )
#     statevector_simulator = bgls.Simulator(
#         statevector_state,
#         bgls.state_vector_bitstring_probability,
#         cirq.protocols.act_on,
#     )
#     statevector_result = statevector_simulator.sample(
#         ghz, repetitions=100, seed=3
#     )
#     density_state = cirq.DensityMatrixSimulationState(
#         qubits=(q0, q1, q2), initial_state=0
#     )
#     density_simulator = bgls.Simulator(
#         density_state,
#         bgls.state_vector_bitstring_probability,
#         cirq.protocols.act_on,
#     )
#     density_result = density_simulator.sample(ghz, repetitions=100, seed=3)
#     assert statevector_result == density_result
