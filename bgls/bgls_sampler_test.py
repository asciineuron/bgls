import pytest
import numpy as np
import cirq
import bgls

"""
Test suite for proper behavior with seeds, measurements, and state types.
"""


def test_seed_continuity():
    # Running our sample function with the same seed should produce
    # identical measurements
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q2], key="result1st"),
    )
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )

    bgls_simulator = bgls.Simulator(
        init_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    bgls_result_1 = bgls_simulator.sample(
        ghz,
        repetitions=100,
        seed=1,
    )
    bgls_result_2 = bgls_simulator.sample(
        ghz,
        repetitions=100,
        seed=1,
    )
    assert bgls_result_1 == bgls_result_2


def test_intermediate_measurements():
    # Intermediate measurement gates do not affect simulation
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
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    bgls_simulator = bgls.Simulator(
        init_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    bgls_result = bgls_simulator.sample(ghz, repetitions=100, seed=1)
    bgls_result_intermediate = bgls_simulator.sample(
        ghz_intermediate, repetitions=100, seed=1
    )
    assert bgls_result == bgls_result_intermediate


def test_multiple_measurements():
    # Distributing final measurement across multiple gates has same behavior
    # as a single measurement of all qubits
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q1, q2], key="result"),
    )
    ghz_multiple = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q2], key="result1st"),
        cirq.measure([q1], key="result2nd"),
    )
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    bgls_simulator = bgls.Simulator(
        init_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    bgls_result = bgls_simulator.sample(ghz, repetitions=100, seed=12)
    bgls_result_multiple = bgls_simulator.sample(
        ghz_multiple, repetitions=100, seed=12
    )
    # converting to histogram for checking recovers original values since
    # spread across multiple measurement keys
    np.testing.assert_array_equal(
        cirq.vis.get_state_histogram(bgls_result),
        cirq.vis.get_state_histogram(bgls_result_multiple),
    )


def test_no_measurements():
    # Sampling a circuit without final measurement raises a ValueError
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q0, q2))
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    bgls_simulator = bgls.Simulator(
        init_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    with pytest.raises(ValueError) as error_info:
        bgls_result = bgls_simulator.sample(
            ghz,
            repetitions=100,
        )


def test_partial_measurements():
    # Measuring only some final qubits yields the same distribution,
    # but with bitstring int rep matching the subset of measured bits
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q2], key="result"),
    )
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    bgls_simulator = bgls.Simulator(
        init_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    bgls_result = bgls_simulator.sample(
        ghz,
        repetitions=100,
    )
    # here because ghz, still only get low and high-end values with equal
    # prob, but now 0 and 3 (ie 00 11) rather than 0 and 7 (000 111)
    # can assert these are the only measured values
    assert set(bgls_result.histogram(key="result")).issubset({0, 3})


def test_density_state():
    # Density matrix simulation yields the same result as state vector
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q1, q2], key="result"),
    )
    statevector_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    statevector_simulator = bgls.Simulator(
        statevector_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    statevector_result = statevector_simulator.sample(
        ghz, repetitions=100, seed=3
    )
    density_state = cirq.DensityMatrixSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    density_simulator = bgls.Simulator(
        density_state,
        bgls.state_vector_bitstring_probability,
        cirq.protocols.act_on,
    )
    density_result = density_simulator.sample(ghz, repetitions=100, seed=3)
    assert statevector_result == density_result
