import pytest
import numpy as np
import cirq
import matplotlib.pyplot as plt
from bgls.module import bgls_sampler
from bgls.module import bgls_utils


def test_intermediate_measurements():
    # Intermediate measurement gates do not affect simulation
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
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
    bgls_result = bgls_sampler.sample(
        ghz,
        init_state,
        bgls_utils.state_vector_bitstring_amplitude,
        cirq.protocols.act_on,
        repetitions=100,
    )
    _ = cirq.plot_state_histogram(bgls_result, plt.subplot())
    plt.show()


def test_multiple_measurements():
    # Distributing final measurement across multiple gates has same behavior
    # as a single measurement of all qubits
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure([q0, q2], key="result1st"),
        cirq.measure([q1], key="result2nd"),
    )
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    bgls_result = bgls_sampler.sample(
        ghz,
        init_state,
        bgls_utils.state_vector_bitstring_amplitude,
        cirq.protocols.act_on,
        repetitions=100,
    )
    _ = cirq.plot_state_histogram(bgls_result, plt.subplot())
    plt.show()


def test_no_measurements():
    # Sampling a circuit without final measurement raises a ValueError
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q0, q2))
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    with pytest.raises(ValueError) as error_info:
        bgls_result = bgls_sampler.sample(
            ghz,
            init_state,
            bgls_utils.state_vector_bitstring_amplitude,
            cirq.protocols.act_on,
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
        cirq.measure([q0, q2], key="result1st"),
    )
    init_state = cirq.StateVectorSimulationState(
        qubits=(q0, q1, q2), initial_state=0
    )
    bgls_result = bgls_sampler.sample(
        ghz,
        init_state,
        bgls_utils.state_vector_bitstring_amplitude,
        cirq.protocols.act_on,
        repetitions=100,
    )
    # here because ghz, still only get low and high-end values with equal
    # prob, but now 0 and 3 (ie 00 11) rather than 0 and 7 (000 111)
    _ = cirq.plot_state_histogram(bgls_result, plt.subplot())
    plt.show()
