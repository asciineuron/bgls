# this will give demonstrations on how our sampler is to be used
# as well as providing clear development targets to satisfy

import cirq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from module import bgls_sampler

# create GHZ circuit
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q0, q2),
    # works equally well since all we need is a single measurement of all
    # qubits:
    cirq.measure([q0, q1, q2], key="result"),
)

# how to sample measurements with cirq  default:
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.run(circuit, repetitions=3)
# results as a list of measurements per qubit eg
# q0rep1 q0rep2 q0rep3, q1rep1 q1rep2 q1rep3, q2rep1 q2rep2 q2rep3
print(cirq_results)

# for now, we are not dealing with measurement gates in the circuit
q0, q1, q2 = cirq.LineQubit.range(3)
circuit_sans_measure = cirq.Circuit(
    cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2)
)
bitstring = bgls_sampler.bgls_sample(cirq.Simulator(), circuit_sans_measure)
print(bitstring)

# run over several repetitions, make histogram to examine correctness:
# should be about equal occurrences of 0 and 1
bgls_results = []
trials = 20
for i in range(trials):
    bgls_results.append(
        bgls_sampler.bgls_sample(cirq.Simulator(), circuit_sans_measure)
    )
bitstrings, counts = np.unique(bgls_results, return_counts=True)
plt.bar(range(len(bitstrings)), counts, align="center")
plt.xticks(range(len(bitstrings)), bitstrings)
plt.show()

# extend over multiple trials, should approach gaussian e.g. coin toss
q0, q1, q2 = cirq.LineQubit.range(3)
ghz = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2))
repetitions = 10
trials = 10
trials_results = []
for t in range(trials):
    bgls_results = []
    for i in range(repetitions):
        bgls_results.append(bgls_sampler.bgls_sample(cirq.Simulator(), ghz))
    bitstrings, counts = np.unique(bgls_results, return_counts=True)
    trials_results.append(counts[0] / repetitions)
pd.Series(trials_results).plot.kde()
plt.show()
