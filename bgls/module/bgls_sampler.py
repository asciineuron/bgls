import itertools
import cirq
import numpy as np
from typing import TypeVar, Callable

State = TypeVar("State")
Bitstring = TypeVar("Bitstring")


def bgls_sample(
        circuit: cirq.Circuit,
        initial_state: State,
        compute_amplitude: Callable[
            [State, Bitstring], complex],
        apply_gate: Callable[
            [cirq.Operation, State], None] = cirq.protocols.act_on
) -> Bitstring:
    rng = np.random.RandomState()

    resolved_circuit = cirq.resolve_parameters(circuit, cirq.ParamResolver({}))
    cirq.sim.simulator.check_all_resolved(resolved_circuit)

    qubits = resolved_circuit.all_qubits()
    qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
    bitstring = "0" * len(qubits)

    state = initial_state
    for op in resolved_circuit.all_operations():
        # apply gate to system:
        apply_gate(op, state)

        # Determine the candidate bitstrings to sample:
        op_support = {qubit_index[q] for q in op.qubits}
        candidates = list(
            itertools.product(
                *[
                    ["0", "1"] if i in op_support else [b]
                    for i, b in enumerate(bitstring)
                ]
            )
        )

        # Compute probability of each candidate bitstring:
        candidate_amplitudes = [compute_amplitude(state, "".join(candidate))
                                for candidate in candidates]
        candidate_probs = np.abs(np.asarray(candidate_amplitudes)) ** 2

        # sample to get bitstring
        bitstring = "".join(
            candidates[
                rng.choice(
                    range(len(candidates)),
                    p=candidate_probs / sum(candidate_probs),
                )
            ]
        )
    return bitstring
