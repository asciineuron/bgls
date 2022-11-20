import itertools
import cirq
import numpy as np


def bgls_sample(
        simulator: cirq.SimulatesAmplitudes,
        circuit: "cirq.AbstractCircuit",
        seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
) -> str:
    """
    Takes any simulator capable of computing bitstring amplitudes over a
    circuit, returns a sampled bitstring.
    """
    rng = np.random.RandomState(seed)
    resolved_circuit = cirq.resolve_parameters(circuit, cirq.ParamResolver({}))
    cirq.sim.simulator.check_all_resolved(resolved_circuit)

    qubits = resolved_circuit.all_qubits()
    qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
    bitstring = "0" * len(qubits)

    subcircuit = cirq.Circuit()
    for op in resolved_circuit.all_operations():
        # compute_amplitudes() works on entire circuit, so must iteratively
        # build up subcircuit to run on
        subcircuit.append(op)

        # Determine the candidate bitstrings to sample.
        op_support = {qubit_index[q] for q in op.qubits}
        candidates = list(
            itertools.product(
                *[
                    ["0", "1"] if i in op_support else [b]
                    for i, b in enumerate(bitstring)
                ]
            )
        )
        # need to convert candidates to sequence of ints (ie turn our
        # bitstrings into their base10 int representations)
        candidates_as_int_list = []
        for candidate in candidates:
            int_rep = int("".join(candidate), 2)
            candidates_as_int_list.append(int_rep)

        # Compute probability of each candidate bitstring.
        # have to make sure all qubits are recognized even if not operating on
        # yet, so pass qubit order of entire (not sub-) circuit
        candidate_amplitudes = simulator.compute_amplitudes(
            subcircuit,
            candidates_as_int_list,
            qubit_order=resolved_circuit.all_qubits(),
        )
        # this is a list of complex coefficients, probs are these squared
        candidate_probs = np.abs(np.asarray(candidate_amplitudes)) ** 2

        bitstring = "".join(
            candidates[
                rng.choice(
                    range(len(candidates)),
                    p=candidate_probs / sum(candidate_probs),
                )
            ]
        )
    return bitstring
