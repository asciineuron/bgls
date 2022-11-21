import pytest
import numpy as np
import cirq
from scipy.stats import kstest
from bgls.module import bgls_sampler


def test_ghz():
    q0, q1, q2 = cirq.LineQubit.range(3)
    ghz = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2))
    repetitions = 10
    trials = 10
    trials_results = []
    for t in range(trials):
        bgls_results = []
        for i in range(repetitions):
            bgls_results.append(
                bgls_sampler.bgls_sample(cirq.Simulator(), ghz)
            )
        bitstrings, counts = np.unique(bgls_results, return_counts=True)
        # only possible outputs are 0 and 1 for non-noisy circuit:
        assert all((b == "000" or b == "111") for b in bitstrings)
        trials_results.append(counts[0] / repetitions)
    # trials_results should be close to gaussian centered about 0.5
    # note, repetitions and trials should be scaled up
    assert kstest(trials_results, "norm")[1] >= 0.05
