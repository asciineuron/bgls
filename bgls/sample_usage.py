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
init_state = cirq.DensityMatrixSimulationState(
    qubits=(q0, q1, q2), initial_state=0
)

# how to sample measurements with bgls
bgls_simulator = bgls_sampler.Simulator(
    init_state,
    bgls_utils.density_matrix_bitstring_probability,
    cirq.protocols.act_on,
)
bgls_results = bgls_simulator.sample(circuit, repetitions=100)
print(bgls_results)
_ = cirq.plot_state_histogram(bgls_results, plt.subplot())
plt.show()

# for a single sampling, can get raw bitstring:
bitstring = bgls_simulator.sample_core(
    circuit,
    rng=cirq.value.parse_random_state(None),
)
print(bitstring)
