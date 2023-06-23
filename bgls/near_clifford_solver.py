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


def ith_bit(num: int, i: int) -> int:
    return (num >> i) & 1


def circuit_clifford_decomposition(
    circuit: cirq.Circuit, fidelity: float = 1.0
) -> Tuple[List[cirq.Circuit], List[complex]]:
    """
    Returns an expansion of a Clifford+T+Rz circuit as a list of pure Clifford
    circuits ideally approximating it, as well as their amplitudes.

    Args:
        circuit: Clifford+T circuit to be expanded.
        fidelity: Fraction of terms in the expansion to keep.

    """

    clifford_circuits = []
    circuit_amplitudes = []

    non_clifford_moment_pos = []
    num_non_clifford = 0
    for i, mom in enumerate(circuit.moments):
        for op in mom.operations:
            if not cirq.has_stabilizer_effect(op):
                assert isinstance(op.gate, cirq.ops.common_gates.ZPowGate)
                num_non_clifford += 1
                non_clifford_moment_pos.append((i, op, op.qubits[0]))

    if num_non_clifford == 0:
        return [circuit], [1.0 + 0.0j]
    else:
        for i in range(int(fidelity * 2**num_non_clifford)):
            circuit_expand = circuit.copy()
            amplitude = 1.0 + 0.0j
            for pos, op, qubit in non_clifford_moment_pos:
                # determine amplitudes based on gate exponent:
                theta = np.pi * op.gate.exponent
                # normalizing s.t. Rz gates have 0 relative phase
                phase = np.exp(
                    (0 + 1j) * theta * (0.5 - op.gate.global_shift)
                )
                amplitude *= phase

                if ith_bit(pos, i):
                    amplitude *= np.cos(theta / 2) - np.sin(theta / 2)
                    circuit_expand.batch_replace([(pos, op, cirq.I(qubit))])
                else:
                    amplitude *= (
                        np.sqrt(2.0)
                        * np.exp(-(0 + 1j) * np.pi / 4)
                        * np.sin(theta / 2)
                    )
                    circuit_expand.batch_replace([(pos, op, cirq.S(qubit))])

            clifford_circuits.append(circuit_expand)
            circuit_amplitudes.append(amplitude)

    return clifford_circuits, circuit_amplitudes
