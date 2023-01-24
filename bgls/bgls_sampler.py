import itertools
import cirq
import numpy as np
from typing import TypeVar, Callable, Any, Dict, Sequence, List


class Simulator:
    State = TypeVar("State")

    def __init__(
            self,
            initial_state: State,
            compute_probability: Callable[[State, str], float],
            apply_gate: Callable[
                [cirq.Operation, State], Any
            ] = cirq.protocols.act_on,  # type: ignore[assignment]
    ):
        self.initial_state = initial_state
        self.compute_probability = compute_probability
        self.apply_gate = apply_gate

    def sample(
            self,
            circuit: cirq.Circuit,
            repetitions: int = 1,
            seed: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
    ) -> cirq.Result:
        rng = cirq.value.parse_random_state(seed)

        resolved_circuit = cirq.resolve_parameters(
            circuit, cirq.ParamResolver({})
        )
        cirq.sim.simulator.check_all_resolved(resolved_circuit)

        if not circuit.are_any_measurements_terminal():
            raise ValueError(
                "Circuit has no terminal measurements to sample."
            )

        records: Dict[str, np.ndarray] = {}

        for rep in range(repetitions):
            keys_to_bitstrs = self.sample_core(
                resolved_circuit,
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

        return cirq.study.ResultDict(records=records)

    def sample_core(
            self,
            circuit: cirq.Circuit,
            rng: np.random.RandomState = cirq.value.parse_random_state(None),
            return_history: bool = False,
    ) -> Dict[str, List[str]]:
        keys_to_indices: Dict[str, List[int]] = {}
        keys_to_bitstrings: Dict[str, List[str]] = {}

        qubits = circuit.all_qubits()
        qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
        bitstring = "0" * len(qubits)
        bitstrings = [bitstring]

        state = self.initial_state.copy()
        for i, op in enumerate(circuit.all_operations()):
            if cirq.protocols.is_measurement(op):
                # check if terminal:
                if circuit.next_moment_operating_on(op.qubits, i + 1) is None:
                    meas_key = cirq.protocols.measurement_key_name(op.gate)
                    if meas_key not in keys_to_indices:
                        meas_indices = [qubit_index[q] for q in op.qubits]
                        keys_to_indices[meas_key] = meas_indices
                continue

            self.apply_gate(op, state)

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
            candidate_probs = np.asarray([
                self.compute_probability(state, candidate)
                for candidate in joined_cands
            ])

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
