import itertools
import cirq
import numpy as np
from typing import TypeVar, Callable, Any, Dict
from operator import itemgetter

State = TypeVar("State")


def bgls_results_wrapper(
    circuit: cirq.Circuit,
    initial_state: State,
    compute_amplitude: Callable[[State, str], complex],
    apply_gate: Callable[
        [cirq.Operation, State], Any
    ] = cirq.protocols.act_on,
    repetitions: int = 1,
) -> cirq.Result:
    resolved_circuit = cirq.resolve_parameters(
        circuit, cirq.ParamResolver({})
    )
    cirq.sim.simulator.check_all_resolved(resolved_circuit)

    qubits = resolved_circuit.all_qubits()
    qubit_index = {q: i for i, q in enumerate(sorted(qubits))}

    records: Dict[str, np.ndarray] = {}

    # make sure there is some terminal measurement:
    if not circuit.are_any_measurements_terminal():
        raise ValueError("Circuit has no terminal measurements to sample.")

    # determine which qubits associated with which terminal measurement gates:
    qubits_per_measurement = {}
    final_moment = resolved_circuit.moments[-1]
    for op in final_moment.operations:
        if cirq.protocols.is_measurement(op):
            meas_key = op.gate.key
            meas_qubits = op.qubits
            qubits_per_measurement[meas_key] = meas_qubits
            if meas_key not in records:
                # assumes only one instance of this key...
                records[meas_key] = np.zeros(
                    (repetitions, 1, len(meas_qubits))
                )

    for rep in range(repetitions):
        # state mutable so have to make copy across repetitions:
        state = initial_state.copy()
        bitstring = bgls_sample(
            resolved_circuit, state, compute_amplitude, apply_gate
        )
        # next just pack into the appropriate format then done!
        for meas_key in records:
            meas_indices = set()
            for q in qubits_per_measurement[meas_key]:
                meas_indices.add(qubit_index[q])
            # extract relevant bitstring indices, convert from string to array
            records[meas_key][rep, 0, :] = np.fromstring(
                "".join(itemgetter(*meas_indices)(bitstring)), "u1"
            ) - ord("0")

    result = cirq.study.ResultDict(records=records)
    return result


def bgls_sample(
    circuit: cirq.Circuit,
    initial_state: State,
    compute_amplitude: Callable[[State, str], complex],
    apply_gate: Callable[
        [cirq.Operation, State], Any
    ] = cirq.protocols.act_on,
) -> str:
    rng = np.random.RandomState()

    qubits = circuit.all_qubits()
    qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
    bitstring = "0" * len(qubits)

    state = initial_state
    for op in circuit.all_operations():
        # apply gate to system, leaving out measurement for now:
        # (caused wavefunc collapse couldn't apply our algo as is)
        if cirq.protocols.is_measurement(op):
            break

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
        joined_cands = ["".join(cand) for cand in candidates]
        rev_cands = [cand[::-1] for cand in joined_cands]

        # Compute probability of each candidate bitstring:
        candidate_amplitudes = [
            compute_amplitude(state, candidate) for candidate in rev_cands
        ]
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
