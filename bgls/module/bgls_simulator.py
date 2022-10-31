import functools
import itertools
from builtins import set
from typing import Dict, FrozenSet

import cirq
import numpy as np


class BglsSimulator(cirq.sim.SimulatesSamples):
    """General description here.

    Details here.
    """

    def __init__(self, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        self._seed = seed

    def _run(
            self,
            circuit: 'cirq.AbstractCircuit',
            param_resolver: 'cirq.ParamResolver',
            repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or cirq.ParamResolver({})  # in case empty
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        cirq.sim.simulator.check_all_resolved(resolved_circuit)

        return self._ryan_sample_final_results(resolved_circuit, repetitions)

    def _ryan_apply(
            self,
            operation: cirq.Operation,
            state: np.ndarray,
            qubits: FrozenSet[cirq.Qid],
    ) -> np.ndarray:
        unitary = cirq.unitary(
            cirq.Moment(operation, [cirq.I.on(q) for q in qubits - set(operation.qubits)])
        )
        return unitary @ state

    def _ryan_sample_final_results(
            self,
            resolved_circuit: cirq.AbstractCircuit,
            repetitions: int
    ) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(self._seed)
        records: Dict[str, np.ndarray] = {}

        qubits = resolved_circuit.all_qubits()
        qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
        bitstring = "0" * len(qubits)

        for repetition in range(repetitions):
            state = functools.reduce(  # TODO: Make this an input.
                lambda a, b: a * b, [cirq.KET_ZERO(q) for q in sorted(qubits)]
            ).state_vector()

            for op in resolved_circuit.all_operations():
                # record and quit when we hit the measurement gate:
                if isinstance(op.gate, cirq.MeasurementGate):
                    print(op)
                    meas_key = op.gate.key
                    if meas_key not in records:
                        # TODO assume here we only have 1 appearance of key... so won't work with general circuit
                        records[meas_key] = np.zeros((repetitions, 1, len(qubits)))
                    records[meas_key][repetition, 0, :] = 0.1  # TODO make this fit: np.fromstring(bitstring)
                    break

                # Determine the candidate bitstrings to sample.
                op_support = {qubit_index[q] for q in op.qubits}
                candidates = list(itertools.product(
                    *[["0", "1"] if i in op_support  # Update bits on operation support.
                      else [b]  # Fix bits not on operation support.
                      for i, b in enumerate(bitstring)]
                ))

                # Compute probability of each candidate bitstring.
                state = self._ryan_apply(op, state, qubits)  # TODO: Make this an input.
                probs = [abs(state[int("".join(bits), 2)]) ** 2 for bits in candidates]

                # Sample from the candidate bitstrings and update the bitstring.
                bitstring = "".join(
                    candidates[rng.choice(range(len(candidates)), p=probs / sum(probs))]
                )

        return records
