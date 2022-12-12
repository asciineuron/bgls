import cirq
from module import bgls_sampler, bgls_utils
import matplotlib.pyplot as plt

# create GHZ circuit with measurement
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q0, q2),
    cirq.measure([q0, q1, q2], key="result"),
)

# how to sample measurements with cirq default:
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.run(circuit, repetitions=100)
print(cirq_results)
_ = cirq.plot_state_histogram(cirq_results, plt.subplot())
plt.show()

# initialize state for bgls:
init_state = cirq.StateVectorSimulationState(
    qubits=(q0, q1, q2), initial_state=0
)

# for a single sampling, can get raw bitstring:
bitstring = bgls_sampler.sample_core(
    circuit,
    init_state.copy(),
    bgls_utils.compute_state_vector_amplitude,
    cirq.protocols.act_on,
    rng=cirq.value.parse_random_state(None),
)
print(bitstring)

# sampling measurements with bgls, identical to run() output cirq.Result
bgls_result = bgls_sampler.sample(
    circuit,
    init_state,
    bgls_utils.compute_state_vector_amplitude,
    cirq.protocols.act_on,
    repetitions=100,
)
print(bgls_result)
_ = cirq.plot_state_histogram(bgls_result, plt.subplot())
plt.show()
