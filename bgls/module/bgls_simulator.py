from builtins import set

import cirq
import numpy as np
import functools
import itertools
from typing import Dict, Sequence, Set, List, Any, Tuple, FrozenSet


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

    # def _records_to_results(self, records: Dict[str, List]) -> Dict[str, np.ndarray]:
    #     """Takes list of repetition,outcomes per measurement key, converts to numpy array as required for `_run()`."""
    #     results:Dict[str, np.ndarray] = {}
    #     # need to find longest bitstring, luckily they should always be the full length and hence the same?
    #     for meas_key in records.keys():
    #         # len(records[meas_key]) is equal to repetitions ? unless seen gate several times per rep.
    #         # records[meas_key][-1][0] is the rep field of the last entry, should be ok
    #         results[meas_key] = np.zeros((records[meas_key][-1][0],
    #                                       records[meas_key][-1][0]/len(records[meas_key]),
    #                                       len(records[meas_key][-1][1])))
    #         for i in range(records[meas_key][-1][0]):
    #             results[meas_key][i,1,:] = records[meas_key]
    #     return results

    def _ryan_sample_final_results(
            self,
            resolved_circuit: cirq.AbstractCircuit,
            repetitions: int
    ) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(self._seed)
        records: Dict[str, np.ndarray] = {}  # Dict[str, List] = {}
        results: Dict[str, np.ndarray] = {}

        qubits = resolved_circuit.all_qubits()
        qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
        bitstring = "0" * len(qubits)

        for repetition in range(repetitions):
            state = functools.reduce(  # TODO: Make this an input.
                lambda a, b: a * b, [cirq.KET_ZERO(q) for q in sorted(qubits)]
            ).state_vector()

            for op in resolved_circuit.all_operations():
                # record and quit when we hit the measurement gate:
                # first store as list and then convert to np array after all data is collected so no resizing!
                if isinstance(op.gate, cirq.MeasurementGate):  # i.e. is a measurement gate
                    print(op)
                    meas_key = op.gate.key
                    if meas_key not in records:
                        # TODO assume here we only have 1 appearance of key... so won't work with general circuit
                        records[meas_key] = np.zeros((repetitions, 1, len(qubits)))
                        # records[meas_key] = []
                    # records[meas_key].append([repetition, int(bitstring)])  # how to count # times key seen?
                    records[meas_key][repetition, 0, :] = 0.1  # TODO make this fit: np.fromstring(bitstring)
                    # need to break because can't use ryan's apply with measurement gates
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
                # now just to convert bitstring into proper output format,
                # and then wrap in a repetitions loop and return

        return records

    def _sample_final_results(self, resolved_circuit: cirq.AbstractCircuit, repetitions: int) -> Dict[str, np.ndarray]:
        # the above-executed code is needed for any approach, i.e. preprocessing work, whereas this
        # is solution-specific code.
        # let's try the most naive approach first and work up to more efficient/complex solutions
        qubits = tuple(sorted(resolved_circuit.all_qubits()))
        qubits_set = resolved_circuit.all_qubits()
        measurements: Dict[Any, Any] = {}
        for repetition in range(repetitions):
            # from Ryan's colab, set initial state:
            state = functools.reduce(
                lambda a, b: a * b, [cirq.KET_ZERO(q) for q in qubits]
            ).state_vector()
            for op in resolved_circuit.all_operations():
                support = op.qubits  # set of qubits acted on (Ryan instead had the positional numbers?)
                A = tuple(sorted(
                    set(range(1, len(qubits))) - set(sorted(qubits_set))))  # make sure 1 based index is ok
                S: set[cirq.Qid] = set()

        return measurements
