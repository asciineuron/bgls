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
import collections
from typing import TypeVar, Callable, Dict, List

import numpy as np

import cirq

# for now restrict to generic simulationstatebase which is subclassed
State = TypeVar("State", bound=cirq.SimulationStateBase)


def needs_trajectories(
    circuit: "cirq.AbstractCircuit",
) -> bool:
    """Determines if repeated samples can be drawn for a single
    simulation. For near-clifford, noisy, or non-unitary circuits this
    is not possible. Adapted from
    https://github.com/quantumlib/qsim/blob/235ae2fc039fb4a98beb4a6114d10c7f8d2070f7/qsimcirq/qsim_simulator.py#L29.
    """
    if not circuit.are_all_measurements_terminal():
        return True

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


class Simulator(cirq.SimulatesSamples):
    def __init__(
        self,
        initial_state: State,
        apply_op: Callable[
            [cirq.Operation, State], None
        ],  # TODO: Decide on return value.
        compute_probability: Callable[[State, str], float],
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> None:
        """Initialize a BGLS Simulator.

        Args:
            initial_state: The initial state of the circuit (conventionally the
                all |0⟩ state.) Note: Must be provided to indicate the type and
                 how to apply operations, in accordance with `apply_op` below.
            apply_op: Function which inputs an operation and state and
                applies the operation to the state, updating the state in
                place.
            compute_probability: Function which inputs a state and bitstring
                and returns the probability of the given bitstring.
            seed: Seed or random state for sampling.
        """
        # TODO: These three parameters (initial_state, apply_op,
        #  and compute_probability) are not independent and should be handled
        #  by a data class, e.g., `BglsOptions`.

        # TODO: Add default options.
        self._initial_state = initial_state
        self._apply_op = apply_op
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

        If the circuit is unitary, all repetitions are sampled in one pass through
        the circuit. Else, multiple passes are performed.

        Args:
            circuit: The circuit to simulate.
            repetitions: The number of times to simulate the circuit
                (number of measurements to return).
        """
        records: Dict[str, np.ndarray] = {}
        keys_to_bitstrings_list = []

        if not needs_trajectories(circuit):
            # Sample all bitstrings in one pass through the circuit.
            keys_to_bitstrings_list = (
                self._sample_from_one_wavefunction_evolution(
                    circuit, repetitions
                )
            )
        else:
            # Sample one bitstring per trajectory.
            for _ in range(repetitions):
                keys_to_bitstrings_list.append(
                    self._sample_from_one_wavefunction_evolution(
                        circuit, repetitions=1
                    )[0]
                )

        # Format sampled bitstrings.
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

    def _sample_from_one_wavefunction_evolution(
        self, circuit: "cirq.AbstractCircuit", repetitions: int = 1
    ) -> List[Dict[str, List[str]]]:
        """Returns bitstrings sampled in parallel via one pass through the circuit.

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

        bitstrings = {"0" * len(qubits): repetitions}

        state = (
            self._initial_state.copy()
        )  # TODO: Update or require states to have copy method.
        keys_to_indices: Dict[str, List[int]] = {}

        for i, moment in enumerate(circuit.moments):
            for op in moment.operations:
                # Store keys of terminal measurements.
                if (
                    cirq.protocols.is_measurement(op)
                    and circuit.next_moment_operating_on(op.qubits, i + 1)
                    is None
                ):
                    meas_key = cirq.protocols.measurement_key_name(op.gate)
                    if meas_key not in keys_to_indices:
                        meas_indices = [qubit_index[q] for q in op.qubits]
                        keys_to_indices[meas_key] = meas_indices
                    continue

                self._apply_op(op, state)

                # Skip updating bitstrings for diagonal gates since they do not change
                # the probability distribution.
                if all(cirq.is_diagonal(kraus) for kraus in cirq.kraus(op)):
                    continue

                # Memoize self._compute_probability.
                computed_probabilities: Dict[str, float] = {}

                def compute_probability(
                    wavefunction: State, bstring: str
                ) -> float:
                    if bstring in computed_probabilities.keys():
                        return computed_probabilities[bstring]
                    probability = self._compute_probability(
                        wavefunction, bstring
                    )
                    computed_probabilities[bstring] = probability
                    return probability

                # Update bits on support of this operation.
                new_bitstrings: Dict[str, int] = collections.defaultdict(int)
                op_support = {qubit_index[q] for q in op.qubits}
                for bitstring, count in bitstrings.items():
                    candidates = list(
                        itertools.product(
                            *[
                                ["0", "1"] if i in op_support else [b]
                                for i, b in enumerate(bitstring)
                            ]
                        )
                    )

                    # Compute probability of each candidate bitstring.
                    probabilities = np.array(
                        [
                            compute_probability(state, "".join(candidate))
                            for candidate in candidates
                        ]
                    )

                    # Sample new bitstring(s).
                    new_bitstring_indices = self._rng.choice(
                        len(candidates),
                        p=probabilities / sum(probabilities),
                        size=count,
                    )
                    for new_bitstring_index in new_bitstring_indices:
                        new_bitstrings[candidates[new_bitstring_index]] += 1

                bitstrings = new_bitstrings

        # Unflatten for conversion to cirq.Result.
        samples = []
        for bitstring, count in bitstrings.items():
            for _ in range(count):
                samples.append(bitstring)

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
                            for i, bit in enumerate(samples[rep])
                            if i in keys_to_indices[meas]
                        ]
                    )
                ]
        return keys_to_bitstrings
