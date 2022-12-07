import itertools
import cirq
import numpy as np
from typing import TypeVar, Callable, Any, Dict, Sequence
from operator import itemgetter

State = TypeVar("State")


def sample(
    circuit: cirq.Circuit,
    initial_state: State,
    compute_amplitude: Callable[[State, str], complex],
    apply_gate: Callable[
        [cirq.Operation, State], Any
    ] = cirq.protocols.act_on,
    repetitions: int = 1,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
) -> cirq.Result:
    resolved_circuit = cirq.resolve_parameters(
        circuit, cirq.ParamResolver({})
    )
    cirq.sim.simulator.check_all_resolved(resolved_circuit)

    if not circuit.are_any_measurements_terminal():
        raise ValueError("Circuit has no terminal measurements to sample.")

    records: Dict[str, np.ndarray] = {}

    for rep in range(repetitions):
        record = sample_core(
            resolved_circuit,
            initial_state,
            compute_amplitude,
            apply_gate,
            seed,
        )
        for meas_key in record:
            if rep == 0 and meas_key not in records:
                records[meas_key] = np.zeros(
                    (repetitions, 1, np.size(record[meas_key], 1))
                )
            records[meas_key][rep, 0, :] = record[meas_key][0, :]

    result = cirq.study.ResultDict(records=records)
    return result


def sample_core(
    circuit: cirq.Circuit,
    initial_state: State,
    compute_amplitude: Callable[[State, str], complex],
    apply_gate: Callable[
        [cirq.Operation, State], Any
    ] = cirq.protocols.act_on,
    seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
    return_as_bitstring: bool = False,
    return_history: bool = False,
):
    # TODO how to combine return_history with returning records?
    rng = cirq.value.parse_random_state(seed)
    records: Dict[str, np.ndarray] = {}

    qubits = circuit.all_qubits()
    qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
    bitstring = "0" * len(qubits)
    bitstrings = [bitstring]

    state = initial_state.copy()
    for i, op in enumerate(circuit.all_operations()):
        if cirq.protocols.is_measurement(op):
            # check if terminal:
            if circuit.next_moment_operating_on(op.qubits, i + 1) is None:
                meas_key = op.gate.key
                if meas_key not in records:
                    # might have bug if multiple measurements with same
                    # key but different qubits
                    records[meas_key] = np.zeros((1, len(op.qubits)))
                    # convert measured subset of bitstring to binary int rep:
                    meas_indices = [qubit_index[q] for q in op.qubits]
                    records[meas_key][0, :] = np.fromstring(
                        "".join(itemgetter(*(meas_indices))(bitstring)), "u1"
                    ) - ord("0")
            continue

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

        # Compute probability of each candidate bitstring:
        candidate_amplitudes = [
            compute_amplitude(state, candidate) for candidate in joined_cands
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
        bitstrings.append(bitstring)

    if return_as_bitstring:
        if return_history:
            return bitstring, bitstrings
        else:
            return bitstring
    else:
        if return_history:
            return records, bitstrings
        else:
            return records
