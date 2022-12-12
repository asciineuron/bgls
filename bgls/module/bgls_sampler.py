import itertools
import cirq
import numpy as np
from typing import TypeVar, Callable, Any, Dict, Sequence, List

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
    rng = cirq.value.parse_random_state(seed)

    resolved_circuit = cirq.resolve_parameters(
        circuit, cirq.ParamResolver({})
    )
    cirq.sim.simulator.check_all_resolved(resolved_circuit)

    if not circuit.are_any_measurements_terminal():
        raise ValueError("Circuit has no terminal measurements to sample.")

    records: Dict[str, np.ndarray] = {}

    for rep in range(repetitions):
        keys_to_bitstrs = sample_core(
            resolved_circuit,
            initial_state,
            compute_amplitude,
            apply_gate,
            rng=rng,
        )
        for meas_key in keys_to_bitstrs:
            if rep == 0 and meas_key not in records:
                records[meas_key] = np.zeros(
                    (repetitions, 1, len(keys_to_bitstrs[meas_key][-1]))
                )
            records[meas_key][rep, 0, :] = [
                int(bit) for bit in keys_to_bitstrs[meas_key][-1]
            ]

    result = cirq.study.ResultDict(records=records)
    return result


def sample_core(
    circuit: cirq.Circuit,
    initial_state: State,
    compute_amplitude: Callable[[State, str], complex],
    apply_gate: Callable[
        [cirq.Operation, State], Any
    ] = cirq.protocols.act_on,
    rng: np.random.RandomState = None,
    return_history: bool = False,
) -> Dict[str, List[str]]:
    keys_to_indices: Dict[str, List[int]] = {}
    keys_to_bitstrings: Dict[str, List[str]] = {}

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
                if meas_key not in keys_to_indices:
                    meas_indices = [qubit_index[q] for q in op.qubits]
                    keys_to_indices[meas_key] = meas_indices
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

    # return dict of list of bitstrings measured per gate
    # (optionally over time)
    for meas in keys_to_indices:
        if not return_history:
            keys_to_bitstrings[meas] = [
                "".join(
                    [
                        bit
                        for i, bit in enumerate(bitstring)
                        if i in keys_to_indices[meas]
                    ]
                )
            ]
        else:
            keys_to_bitstrings[meas] = [
                "".join(
                    [
                        bit
                        for i, bit in enumerate(bitstr)
                        if i in keys_to_indices[meas]
                    ]
                )
                for bitstr in bitstrings
            ]

    return keys_to_bitstrings
