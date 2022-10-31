# this will give demonstrations on how our sampler is to be used
# as well as providing clear development targets to satisfy

import cirq
from module import bgls_simulator

# create GHZ circuit
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    # works equally well since all we need is a single measurement of all
    # qubits:
    cirq.measure([q0, q1, q2], key="result"),
)

# how to sample measurements with cirq  default:
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.run(circuit, repetitions=3)
print(cirq_results)

# how to sample with our sampler:
bgls_simulator = bgls_simulator.BglsSimulator()
bgls_results = bgls_simulator.run(circuit, repetitions=2)
print(bgls_results)
