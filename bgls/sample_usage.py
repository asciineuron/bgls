# this will give demonstrations on how our sampler is to be used
# as well as providing clear development targets to satisfy

import cirq
from module import bgls_sampler, bgls_utils
import matplotlib.pyplot as plt

# create GHZ circuit
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

# need inverted qubit ordering to handle state vector properly
init_state = cirq.StateVectorSimulationState(
    qubits=(q0, q1, q2), initial_state=0
)  #
# cirq.StateVectorSimulationState(qubits=(q2, q1, q0))

# for a single sampling:
bitstring = bgls_sampler.bgls_sample(
    circuit,
    init_state.copy(),
    bgls_utils.state_vector_bitstring_amplitude,
    cirq.protocols.act_on,
)
print(bitstring)

# wrapping sampling in a cirq.Result across repetitions:
bgls_result = bgls_sampler.bgls_results_wrapper(
    circuit,
    init_state,
    bgls_utils.state_vector_bitstring_amplitude,
    cirq.protocols.act_on,
    repetitions=100,
)
print(bgls_result)
_ = cirq.plot_state_histogram(bgls_result, plt.subplot())
plt.show()
