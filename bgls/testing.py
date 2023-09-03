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

"""Tools for unit testing."""

from typing import Union, Sequence, Optional, Set, List

import cirq


def generate_random_circuit(
    qubits: Union[Sequence[cirq.ops.Qid], int],
    n_moments: int,
    op_density: float,
    gate_domain: Optional[Set[cirq.Gate]] = None,
    random_state: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None,
) -> cirq.circuits.Circuit:
    """Generates a random circuit.

    Args:
        qubits: If a sequence of qubits, then these are the qubits that
            the circuit should act on. Because the qubits on which an
            operation acts are chosen randomly, not all given qubits
            may be acted upon. If an int, then this number of qubits will
            be automatically generated, and the qubits will be
            `cirq.NamedQubits` with names given by the integers in
            `range(qubits)`.
        n_moments: The number of moments in the generated circuit.
        op_density: The probability that a gate is selected to operate on
            randomly selected qubits. Note that this is not the expected number
            of qubits that are acted on, since there are cases where the
            number of qubits that a gate acts on does not evenly divide the
            total number of qubits.
        gate_domain: The set of gates to choose from, specified as a set of
        gates. If not provided, the default gate
            domain is
            {X, Y, Z, H, S, T, CNOT, CZ, SWAP, ISWAP, CZPowGate()}. Only gates
            which act on a number of qubits less than len(qubits) (or qubits if
            provided as an int) are selected from the gate domain.
        random_state: Random state or random state seed.

    Raises:
        ValueError:
            * op_density is not in (0, 1].
            * gate_domain is empty.
            * qubits is an int less than 1 or an empty sequence.

    Returns:
        The randomly generated Circuit.
    """
    if not 0 < op_density <= 1:
        raise ValueError(f"op_density must be in (0, 1] but was {op_density}.")
    if gate_domain is None:
        gate_domain = {
            cirq.X,
            cirq.Y,
            cirq.Z,
            cirq.H,
            cirq.S,
            cirq.T,
            cirq.CNOT,
            cirq.CZ,
            cirq.SWAP,
            cirq.ISWAP,
            cirq.CZPowGate(),
        }
    if not gate_domain:
        raise ValueError("gate_domain must be non-empty.")

    if isinstance(qubits, int):
        qubits = tuple(cirq.ops.NamedQubit(str(i)) for i in range(qubits))
    n_qubits = len(qubits)
    if n_qubits < 1:
        raise ValueError("At least one qubit must be specified.")
    gate_domain = {g for g in gate_domain if g.num_qubits() <= n_qubits}
    if not gate_domain:
        raise ValueError(
            f"After removing gates that act on less than "
            f"{n_qubits} qubits, gate_domain had no gates."
        )
    max_arity = max([g.num_qubits() for g in gate_domain])

    prng = cirq.value.parse_random_state(random_state)

    moments: List[cirq.circuits.Moment] = []
    gate_list = sorted(gate_domain, key=repr)
    num_gates = len(gate_domain)
    for _ in range(n_moments):
        operations = []
        free_qubits = set(qubits)
        while len(free_qubits) >= max_arity:
            gate = gate_list[prng.randint(num_gates)]
            op_qubits = prng.choice(
                sorted(free_qubits),
                size=gate.num_qubits(),
                replace=False
                # type: ignore[arg-type]
            )
            free_qubits.difference_update(op_qubits)
            if prng.rand() <= op_density:
                operations.append(gate(*op_qubits))
        moments.append(cirq.circuits.Moment(operations))

    return cirq.circuits.Circuit(moments)
