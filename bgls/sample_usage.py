# this will give demonstrations on how our sampler is to be used
# as well as providing clear development targets to satisfy

import cirq
from bgls.module import bgls_simulator

# create GHZ circuit
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0,q1),
    cirq.CNOT(q1,q2)
)

# how to sample measurements with cirq's default:
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.run(circuit, repetitions=2)

# how to sample with our sampler:
gate_simulator = bgls_simulator.GateSamplerSimulator()
gate_results = bgls_simulator.run(circuit, repetitions=2)