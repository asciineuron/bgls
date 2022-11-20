# this will give demonstrations on how our sampler is to be used
# as well as providing clear development targets to satisfy

import cirq
from module import bgls_sampler

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

# for now, we are not dealing with measurement gates in the circuit
q0, q1, q2 = cirq.LineQubit.range(3)
circuit_sans_measure = cirq.Circuit(
    cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2)
)
bitstring = bgls_sampler.bgls_sample(cirq.Simulator(), circuit_sans_measure)
print(bitstring)
