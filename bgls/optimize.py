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

"""Tools for optimizing circuits for BGLS."""

import cirq


def optimize_for_bgls(circuit: cirq.Circuit) -> cirq.Circuit:
    """Modifies circuit operations to improve efficiency of BGLS simulation.

    Args:
        circuit: The circuit to optimize.

    Returns:
        The optimized circuit.
    """
    if len(list(circuit.all_operations())) == 0:
        return circuit

    circuit = cirq.drop_negligible_operations(circuit)

    max_unitary_support = max(
        [
            len(op.qubits)
            for op in circuit.all_operations()
            if cirq.has_unitary(op)
        ]
    )
    return cirq.merge_k_qubit_unitaries(circuit, k=max_unitary_support)
