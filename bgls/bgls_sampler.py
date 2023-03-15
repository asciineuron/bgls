# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the BGLS Simulator."""

import itertools
from typing import TypeVar, Callable, Dict, List

import numpy as np

import cirq


State = TypeVar("State")


class Simulator(cirq.SimulatesSamples):
    def __init__(
        self,
        initial_state: State,
        apply_gate: Callable[
            [cirq.Operation, State], None
        ],  # TODO: Decide on return value.
        compute_probability: Callable[[State, str], float],
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> None:
        """Initialize a BGLS Simulator.

        Args:
            initial_state: The initial state of the circuit (conventionally the
                all |0⟩ state.) Note: Must be provided to indicate the type and
                 how to apply gates, in accordance with `apply_gate` below.
            apply_gate: Function which inputs an operation and state and
                applies the operation to the state, updating the state in
                place.
            compute_probability: Function which inputs a state and bitstring
                and returns the probability of the given bitstring.
            seed: Seed or random state for sampling.
        """
        # TODO: These three parameters (initial_state, apply_gate,
        #  and compute_probability) are not independent and should be handled
        #  by a data class, e.g., `BglsOptions`.

        # TODO: Add default options.
        self._initial_state = initial_state
        self._apply_gate = apply_gate
        self._compute_probability = compute_probability

        self._rng = cirq.value.parse_random_state(seed)

    def _run(
        self,
        circuit: "cirq.AbstractCircuit",
        param_resolver: "cirq.ParamResolver",
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """Run a simulation, mimicking quantum hardware.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: Number of times to repeat the run.

        Returns:
            A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 3-dimensional
            numpy array, the first dimension corresponding to the
            repetition, the second to the instance of that key in the
            circuit, and the third to the actual boolean measurement
            results (ordered by the qubits being measured.)
        """
        if not circuit.are_any_measurements_terminal():
            raise ValueError("No terminal measurements to sample.")

        param_resolver = param_resolver or cirq.ParamResolver()
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)

        return self._sample(resolved_circuit, repetitions)

    def _sample(
        self,
        circuit: "cirq.AbstractCircuit",
        repetitions: int = 1,
    ) -> Dict[str, np.ndarray]:
        records: Dict[str, np.ndarray] = {}
        """Returns a number of measurements by simulating the circuit.

        Args:
            circuit: The circuit to simulate.
            repetitions: The number of times to simulate the circuit
                (number of measurements to return).
        """
        for rep in range(repetitions):
            keys_to_bitstrings = self._sample_once(circuit)
            for meas_key in keys_to_bitstrings:
                if rep == 0 and meas_key not in records:
                    records[meas_key] = np.zeros(
                        (repetitions, 1, len(keys_to_bitstrings[meas_key][-1]))
                    )
                records[meas_key][rep, 0, :] = [
                    int(bit) for bit in keys_to_bitstrings[meas_key][-1]
                ]

        return records

    def _sample_once(
        self, circuit: "cirq.AbstractCircuit"
    ) -> Dict[str, List[str]]:
        """Returns one measurement by simulating the circuit.

        Args:
            circuit: The circuit to simulate.
        """
        qubits = circuit.all_qubits()
        qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
        bitstring = "0" * len(qubits)
        bitstrings = [bitstring]

        state = (
            self._initial_state.copy()
        )  # TODO: Update or require states to have copy method.
        keys_to_indices: Dict[str, List[int]] = {}
        for i, op in enumerate(circuit.all_operations()):
            if cirq.protocols.is_measurement(op):
                if circuit.next_moment_operating_on(op.qubits, i + 1) is None:
                    meas_key = cirq.protocols.measurement_key_name(op.gate)
                    if meas_key not in keys_to_indices:
                        meas_indices = [qubit_index[q] for q in op.qubits]
                        keys_to_indices[meas_key] = meas_indices
                continue

            self._apply_gate(op, state)

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
            joined_cands = ["".join(cand) for cand in candidates]

            # Compute probability of each candidate bitstring.
            candidate_probs = np.asarray(
                [
                    self._compute_probability(state, candidate)
                    for candidate in joined_cands
                ]
            )

            # Sample to get bitstring.
            bitstring = "".join(
                candidates[
                    self._rng.choice(
                        range(len(candidates)),
                        p=candidate_probs / sum(candidate_probs),
                    )
                ]
            )
            bitstrings.append(bitstring)

        # Return dict of list of bitstrings measured per gate.
        keys_to_bitstrings: Dict[str, List[str]] = {}
        for meas in keys_to_indices:
            keys_to_bitstrings[meas] = [
                "".join(
                    [
                        bit
                        for i, bit in enumerate(bitstring)
                        if i in keys_to_indices[meas]
                    ]
                )
            ]
        return keys_to_bitstrings
