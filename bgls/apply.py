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

"""Tools for applying operatings to states."""

import numpy as np

import cirq

import bgls


def act_on_near_clifford(
    op: cirq.Operation,
    state: bgls.simulator.State,
    rng: np.random.RandomState = np.random.RandomState(),
) -> None:
    """
    Applies a Clifford+T (or more generically, Rz(theta)) operation to a state.
    If the operation is Clifford, apply normally,
    else choose one of the operations in the Clifford-expansion to apply.

    Parameters:
        op: operation to apply.
        state: simulator state to update.
        rng: rng for controlling gate-expansion decisions.
    """
    if cirq.has_stabilizer_effect(op):
        cirq.protocols.act_on(op, state)
    elif isinstance(op.gate, cirq.ops.common_gates.ZPowGate):
        # note this includes Rz gates
        theta = np.pi * op.gate.exponent

        probs = np.power(
            np.abs(
                [
                    np.cos(theta / 2) - np.sin(theta / 2),
                    np.sqrt(2.0)
                    * np.exp(-(0 + 1j) * np.pi / 4)
                    * np.sin(theta / 2),
                ]
            ),
            2,
        )
        cand_gates = [cirq.I(op.qubits[0]), cirq.S(op.qubits[0])]
        chosen_gate = rng.choice(cand_gates, p=probs / sum(probs))
        cirq.protocols.act_on(chosen_gate, state)
    else:
        raise ValueError(
            "Operation must be Clifford, T, or a z rotation. Was called on "
            "the gate " + str(op.gate) + "."
        )
