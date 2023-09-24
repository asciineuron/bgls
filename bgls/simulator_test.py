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

import numpy as np

import cirq

import quimb.tensor as qtn
import cirq.contrib.quimb.mps_simulator

import bgls


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
    M_subset = []
    for i, Ai in enumerate(mps.M):
        qubit_index = mps.i_str(i)
        # selecting the component with matching bitstring:
        A_subset = Ai.isel({qubit_index: int(bitstring[i])})
        M_subset.append(A_subset)

    tensor_network = qtn.TensorNetwork(M_subset)
    bitstring_amplitude = tensor_network.contract(inplace=False)
    return np.power(np.abs(bitstring_amplitude), 2)


@pytest.mark.parametrize("nqubits", range(3, 8 + 1))
def test_samples_correct_bitstrings_for_ghz_circuit(nqubits: int):
    """Tests correct measurement results for a GHZ circuit (should only
    return
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
        apply_op=cirq.protocols.act_on,
        compute_probability=bgls.born.compute_probability_state_vector,
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
        bgls.born.compute_probability_state_vector,
    )
    sim1 = bgls.Simulator(*sim_params, seed=1)
    sim2 = bgls.Simulator(*sim_params, seed=1)

    result1 = sim1.run(circuit, repetitions=100)
    result2 = sim2.run(circuit, repetitions=100)

    assert result2 == result1


def test_final_states():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H.on(a),
        cirq.CNOT.on(a, b),
        cirq.measure(a, b, key="z"),
    )

    sim = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=(a, b), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
    )
    assert len(sim.final_states) == 0

    _ = sim.run(circuit, repetitions=100)
    assert len(sim.final_states) == 1
    assert np.allclose(sim.final_states[0].target_tensor.flatten(), np.array([1, 0, 0, 1]) / np.sqrt(2))

    sim.clear_final_states()
    assert len(sim.final_states) == 0

    needs_trajectories_circuit = cirq.Circuit(
        cirq.H.on(a),
        cirq.bit_flip(0.01).on_each(a, b),
        cirq.CNOT.on(a, b),
        cirq.measure(a, b, key="z"),
    )
    sim.run(needs_trajectories_circuit, repetitions=10)
    assert len(sim.final_states) == 10

    sim.clear_final_states()
    assert len(sim.final_states) == 0



def test_simulation_with_intermediate_measurements():
    """Test simulation with intermediate measurements."""
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H.on(a),
        cirq.measure(a, key="intermediate"),
        cirq.CNOT.on(a, b),
        cirq.measure(a, b, key="terminal"),
    )

    sim = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=(a, b), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
    )
    result = sim.run(circuit, repetitions=100)
    assert {0, 3}.issuperset(result.histogram(key="terminal").keys())

    # Check to make sure the measurement was applied and we get |00⟩ or |11⟩, not |00⟩ + |11⟩.
    for final_state in sim.final_states:
        assert np.allclose(
            final_state.target_tensor.flatten(), [1, 0, 0, 0]
        ) or np.allclose(final_state.target_tensor.flatten(), [0, 0, 0, 1])


def test_run_with_no_terminal_measurements_raises_value_error():
    """Tests simulating a circuit without terminal measurements raises a
    ValueError.
    """
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))

    sim = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=(q,), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
    )

    with pytest.raises(ValueError):
        sim.run(circuit)

    circuit.append([cirq.measure(q), cirq.H(q)])

    with pytest.raises(ValueError):
        sim.run(circuit)


def test_measure_subset_of_qubits_yields_correct_results():
    """Tests measuring a subset of qubits yields the expected results."""
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q2], key="result"),
    )
    sim = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=(q0, q1, q2), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
    )
    result = sim.run(
        ghz,
        repetitions=100,
    )

    # Should only sample bitstrings 00 and 11 (integers 0 and 3).
    assert set(result.histogram(key="result")).issubset({0, 3})


def test_run_with_density_matrix_simulator():
    """Test sampled bitstrings are same when using a density matrix simulator
    and a statevector simulator.
    """
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.CNOT(a, b),
        cirq.X.on(c),
        cirq.measure([a, b, c], key="z"),
    )
    sim_state_vector = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=(a, b, c), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
        seed=1,
    )
    result_state_vector = sim_state_vector.run(circuit, repetitions=100)
    sim_density_matrix = bgls.Simulator(
        cirq.DensityMatrixSimulationState(qubits=(a, b, c), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_density_matrix,
        seed=1,
    )
    result_density_matrix = sim_density_matrix.run(circuit, repetitions=100)

    assert result_density_matrix == result_state_vector


def test_run_with_stabilizer_ch_simulator():
    """Test sampled bitstrings are same when using a state vector ch form
    simulator and a statevector simulator.
    """
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(a),
        cirq.CNOT(a, b),
        cirq.X.on(c),
        cirq.measure([a, b, c], key="z"),
    )
    sim_state_vector = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=(a, b, c), initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
        seed=1,
    )
    result_state_vector = sim_state_vector.run(circuit, repetitions=100)

    sim_stabilizer_ch = bgls.Simulator(
        cirq.StabilizerChFormSimulationState(
            qubits=(a, b, c), initial_state=0
        ),
        cirq.protocols.act_on,
        bgls.born.compute_probability_stabilizer_state,
        seed=1,
    )
    result_stabilizer_ch = sim_stabilizer_ch.run(circuit, repetitions=100)

    assert result_stabilizer_ch == result_state_vector


def test_remains_clifford():
    """Creating a large random circuit of clifford gates, the simulator
    should remain clifford throughout, so act_on behaves identically to
    act_on_near_clifford.
    """
    a, b, c = cirq.LineQubit.range(3)
    domain = {cirq.H: 1, cirq.CNOT: 2, cirq.S: 1}
    clifford_circuit = cirq.testing.random_circuit(
        [a, b, c], n_moments=100, op_density=0.5, gate_domain=domain
    )
    clifford_circuit = clifford_circuit + cirq.measure([a, b, c], key="z")

    # using act_on, would fail if not strictly clifford
    sim_act_on = bgls.Simulator(
        cirq.StabilizerChFormSimulationState(
            qubits=(a, b, c), initial_state=0
        ),
        cirq.protocols.act_on,
        bgls.born.compute_probability_stabilizer_state,
        seed=1,
    )
    sim_results = sim_act_on.run(clifford_circuit, repetitions=100)
    # expect same results as our act_on_stabilizer
    sim_act_on_stab = bgls.Simulator(
        cirq.StabilizerChFormSimulationState(
            qubits=(a, b, c), initial_state=0
        ),
        bgls.apply.act_on_near_clifford,
        bgls.born.compute_probability_stabilizer_state,
        seed=1,
    )
    sim_results_stab = sim_act_on_stab.run(clifford_circuit, repetitions=100)

    assert sim_results == sim_results_stab


def test_run_with_stabilizer_ch_simulator_near_clifford():
    """Test sampled bitstrings are same when using a state vector ch form
    simulator and a statevector simulator. Using apply_near_clifford_gate
    can work with Clifford+T circuits as well.
    """
    qs = cirq.LineQubit.range(3)
    circuit = bgls.testing.generate_random_circuit(
        qs,
        n_moments=10,
        op_density=0.5,
        gate_domain={cirq.S, cirq.CNOT, cirq.H, cirq.T},
        random_state=1,
    )
    circuit.append(cirq.measure(qs))

    sim_state_vector = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=qs, initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
        seed=1,
    )

    sim_stabilizer_ch = bgls.Simulator(
        cirq.StabilizerChFormSimulationState(qubits=qs, initial_state=0),
        bgls.apply.act_on_near_clifford,
        bgls.born.compute_probability_stabilizer_state,
        seed=1,
    )

    observables = [cirq.Z(i) for i in qs]

    state_vec_observables = sim_state_vector.sample_expectation_values(
        circuit,
        observables=observables,
        num_samples=10000,
        permit_terminal_measurements=True,
    )

    stabilizer_ch_observables = sim_stabilizer_ch.sample_expectation_values(
        circuit,
        observables=observables,
        num_samples=10000,
        permit_terminal_measurements=True,
    )

    assert np.allclose(
        state_vec_observables, stabilizer_ch_observables, atol=1e-2
    )


def test_mps_results_match_state_vec():
    """Test sampled bitstrings are same when using a matrix product state
    simulator and a state vector simulator.
    """

    qs = cirq.LineQubit.range(4)
    circuit = cirq.testing.random_circuit(qs, n_moments=20, op_density=0.5)
    circuit = circuit + cirq.measure(qs, key="z")

    sim_state_vector = bgls.Simulator(
        cirq.StateVectorSimulationState(qubits=qs, initial_state=0),
        cirq.protocols.act_on,
        bgls.born.compute_probability_state_vector,
        seed=1,
    )
    result_state_vector = sim_state_vector.run(circuit, repetitions=100)

    mps_state = cirq.contrib.quimb.MPSState(
        qubits=qs, initial_state=0, prng=np.random.RandomState()
    )
    sim_mps = bgls.Simulator(
        mps_state,
        cirq.protocols.act_on,
        cirq_mps_bitstring_probability,
        seed=1,
    )
    result_mps = sim_mps.run(circuit, repetitions=100)

    assert result_mps == result_state_vector


def test_intermediate_measurement_keys_are_ignored():
    """Tests that only terminal measurements are stored in simulation results."""
    # TODO: Cirq does store intermediate measurement keys.
    #  Decide if we want to match this behavior.
    q = cirq.LineQubit.range(3)
    circuit1 = cirq.Circuit(
        (
            cirq.Moment(cirq.X.on_each(q)),
            cirq.Moment(cirq.X.on_each(q)),
            cirq.Moment(cirq.measure([q[0]], key="q0")),
            cirq.Moment(cirq.X.on(q[0])),
            cirq.Moment(cirq.measure([q[1]], key="q1")),
        )
    )

    circuit2 = cirq.Circuit(
        (
            cirq.Moment(cirq.X.on_each(q)),
            cirq.Moment(cirq.X.on_each(q)),
            cirq.Moment(cirq.X.on(q[0])),
            cirq.Moment(cirq.measure([q[1]], key="q1")),
        )
    )
    sim = bgls.Simulator(
        initial_state=cirq.StateVectorSimulationState(
            qubits=q, initial_state=0
        ),
        apply_op=cirq.protocols.act_on,
        compute_probability=bgls.born.compute_probability_state_vector,
    )

    results1 = sim.run(circuit1, repetitions=100)
    results2 = sim.run(circuit2, repetitions=100)

    assert results1 == results2


@pytest.mark.parametrize("channel", (cirq.depolarize, cirq.amplitude_damp))
def test_simulate_with_noise_common_single_qubit_channels(channel):
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
        apply_op=cirq.protocols.act_on,
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


def test_simulate_with_custom_noise_channel():
    """Tests correctness of simulating circuits with custom
    noise channels.
    """

    class BitAndPhaseFlipChannel(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def __init__(self, p: float) -> None:
            self._p = p

        def _mixture_(self):
            ps = [1.0 - self._p, self._p]
            ops = [cirq.unitary(cirq.I), cirq.unitary(cirq.Y)]
            return tuple(zip(ps, ops))

        def _has_mixture_(self) -> bool:
            return True

        def _circuit_diagram_info_(self, args) -> str:
            return f"BitAndPhaseFlip({self._p})"

    qubits = cirq.LineQubit.range(3)

    circuit = cirq.testing.random_circuit(qubits, n_moments=5, op_density=1)
    circuit = circuit.with_noise(BitAndPhaseFlipChannel(0.01))

    sim = bgls.Simulator(
        initial_state=cirq.StateVectorSimulationState(
            qubits=qubits, initial_state=0
        ),
        apply_op=cirq.protocols.act_on,
        compute_probability=bgls.born.compute_probability_state_vector,
    )
    sim_cirq = cirq.Simulator()

    # Test expectation of observables match Cirq.Simulator.
    observables = [
        cirq.X.on(qubits[0]),
        cirq.Y.on(qubits[1]),
        cirq.Z.on(qubits[2]),
    ]

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
