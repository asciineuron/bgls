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

# for now restrict to generic simulationstatebase which is subclassed
State = TypeVar("State", bound=cirq.SimulationStateBase)


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
        """Returns a number of measurements by simulating the circuit.

        Args:
            circuit: The circuit to simulate.
            repetitions: The number of times to simulate the circuit
                (number of measurements to return).
        """
        records: Dict[str, np.ndarray] = {}
        keys_to_bitstrings_list = []
        if not needs_trajectories(self._apply_gate, circuit):
            keys_to_bitstrings_list = self._perform_bgls_sampling(
                circuit, repetitions
            )
        else:
            for _ in range(repetitions):
                keys_to_bitstrings_list.append(
                    self._perform_bgls_sampling(circuit, 1)[0]
                )

        for rep, keys_to_bitstrings in enumerate(keys_to_bitstrings_list):
            for meas_key in keys_to_bitstrings:
                if rep == 0 and meas_key not in records:
                    records[meas_key] = np.zeros(
                        (
                            repetitions,
                            1,
                            len(keys_to_bitstrings[meas_key][-1]),
                        )
                    )
                records[meas_key][rep, 0, :] = [
                    int(bit) for bit in keys_to_bitstrings[meas_key][-1]
                ]
        return records

    def _perform_bgls_sampling(
        self, circuit: "cirq.AbstractCircuit", repetitions: int = 1
    ) -> List[Dict[str, List[str]]]:
        """Performs the actual bgls sampling algorithm. Updates all
        repetitions of bitstrings in one pass through the circuit.

        Args:
            circuit: The circuit to simulate.
            repetitions: The number of bitstrings to sample from the circuit.

        Returns:
            A list of dictionaries for each bitstring sampled, mapping from
            measurement gate key to the corresponding bitstring subset. Pass
            to _sample to properly format for matching cirq.
        """
        qubits = circuit.all_qubits()
        qubit_index = {q: i for i, q in enumerate(sorted(qubits))}
        bitstring = "0" * len(qubits)
        bitstrings = [bitstring for _ in range(repetitions)]
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

            # Update bits on support of this operation.
            op_support = {qubit_index[q] for q in op.qubits}
            candidates_list = []
            candidate_probs_list = []

            # Memoize self._compute_probability.
            computed_probabilities: Dict[str:float] = {}

            def compute_probability(wavefunction, bstring):
                if bstring in computed_probabilities.keys():
                    return computed_probabilities[bstring]
                probability = self._compute_probability(wavefunction, bstring)
                computed_probabilities[bstring] = probability
                return probability

            # Compute probabilities for all bitstrings.
            for bitstr in bitstrings:
                candidates = list(
                    itertools.product(
                        *[
                            ["0", "1"] if i in op_support else [b]
                            for i, b in enumerate(bitstr)
                        ]
                    )
                )
                candidates_list.append(candidates)

                # Compute probability of each candidate bitstring.
                candidate_probs = np.asarray(
                    [
                        compute_probability(state, "".join(candidate))
                        for candidate in candidates
                    ]
                )
                candidate_probs_list.append(candidate_probs)

            # Sample to get bitstring.
            for rep in range(repetitions):
                bitstrings[rep] = "".join(
                    candidates_list[rep][
                        self._rng.choice(
                            a=range(len(candidates_list[rep])),
                            replace=True,
                            p=candidate_probs_list[rep]
                            / sum(candidate_probs_list[rep]),
                        )
                    ]
                )

        # Return dict of list of bitstrings measured per gate.
        keys_to_bitstrings: List[Dict[str, List[str]]] = [
            {} for _ in range(repetitions)
        ]
        for rep in range(repetitions):
            for meas in keys_to_indices:
                keys_to_bitstrings[rep][meas] = [
                    "".join(
                        [
                            bit
                            for i, bit in enumerate(bitstrings[rep])
                            if i in keys_to_indices[meas]
                        ]
                    )
                ]
        return keys_to_bitstrings


def needs_trajectories(
    apply_gate: Callable[[cirq.Operation, State], None],
    circuit: "cirq.AbstractCircuit",
) -> bool:
    """Determines if repeated samples can be drawn for a single
    simulation. For near-clifford, noisy, or non-unitary circuits this
    is not possible. Taken from
    https://github.com/quantumlib/qsim/blob
    /235ae2fc039fb4a98beb4a6114d10c7f8d2070f7/qsimcirq/qsim_simulator.py
    #L29"""
    if apply_gate != cirq.act_on:
        return False
    for op in circuit.all_operations():
        test_op = (
            op
            if not cirq.is_parameterized(op)
            else cirq.resolve_parameters(
                op, {param: 1 for param in cirq.parameter_names(op)}
            )
        )
        if not (cirq.is_measurement(test_op) or cirq.has_unitary(test_op)):
            return True
    return False
