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

"""Implementation of additional Clifford circuit handling not yet used in
simulator."""

from typing import List, Tuple

import numpy as np

import cirq


def is_clifford_gate(op: cirq.Operation) -> bool:
    """
    More strictly than cirq.has_stabilizer_effect(op), yields whether the
    gate is in the Clifford group,
    only allowing explicitly H, S, CNOT, and Measurement gates.

    Args:
        op: the gate to be checked.
    """

    is_H = (
        isinstance(op.gate, cirq.ops.common_gates.HPowGate)
        and op.gate.exponent == 1
    )
    is_CNOT = (
        isinstance(op.gate, cirq.ops.common_gates.CXPowGate)
        and op.gate.exponent == 1
    )
    is_Phase = (
        isinstance(op.gate, cirq.ops.common_gates.CZPowGate)
        and op.gate.exponent == 0.5
    )
    return is_H or is_CNOT or is_Phase or cirq.protocols.is_measurement(op)


def ith_bit(num: int, i: int) -> int:
    return (num >> i) & 1


def circuit_clifford_decomposition(
    circuit: cirq.Circuit, fidelity: float = 1.0
) -> Tuple[List[cirq.Circuit], List[complex]]:
    """
    Returns an expansion of a Clifford+T circuit as a list of pure Clifford
    circuits ideally approximating it, as well as their amplitudes.

    Args:
        circuit: Clifford+T circuit to be expanded.
        fidelity: Fraction of terms in the expansion to keep.

    """

    clifford_circuits = []
    circuit_amplitudes = []

    non_clifford_pos = []
    num_non_clifford = 0
    for i, op in enumerate(circuit.all_operations()):
        if not is_clifford_gate(op):
            num_non_clifford += 1
            non_clifford_pos.append(i)

    for i in range(int(fidelity * 2**num_non_clifford)):
        circuit_expand = circuit.copy()
        amplitude = 1.0 + 0.0j
        for pos in non_clifford_pos:
            qubits = [q for q in circuit_expand[pos].qubits]
            if ith_bit(i, pos - 1):
                amplitude *= np.cos(np.pi / 8) - np.sin(np.pi / 8)
                circuit_expand[pos] = cirq.I(qubits[0])
            else:
                amplitude *= (
                    np.sqrt(2.0)
                    * np.exp(-(0 + 1j) * np.pi / 4)
                    * np.sin(np.pi / 8)
                )
                circuit_expand[pos] = cirq.S(qubits[0])
        clifford_circuits.append(circuit_expand)
        circuit_amplitudes.append(amplitude)

    return clifford_circuits, circuit_amplitudes
