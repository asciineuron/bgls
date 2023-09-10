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

from typing import Optional, cast, Sequence


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


def _to_unchecked_state_vector(
    state_rep: "cirq.STATE_VECTOR_LIKE",
    num_qubits: Optional[int] = None,
    *,  # Force keyword arguments
    qid_shape: Optional[Sequence[int]] = None,
    dtype: Optional["DTypeLike"] = None,
    atol: float = 1e-7,
) -> np.ndarray:
    """Derived from cirq.to_valid_state_vector. Converts the state_rep to
    ndarray form. Notably does not check validity, for use with non-unitary
    noise gates

    This method is used to support passing in an integer representing a
    computational basis state or a full state vector as a representation of
    a pure state.

    Args:
        state_rep: If an int, the state vector returned is the state vector
            corresponding to a computational basis state. If a numpy array
            this is the full state vector.
        num_qubits: The number of qubits for the state vector. The state_rep
            must be valid for this number of qubits.
        qid_shape: The expected qid shape of the state vector. Specify this
            argument when using qudits.
        dtype: The numpy dtype of the state vector, will be used when creating
            the state for a computational basis state, or validated against if
            state_rep is a numpy array.
        atol: Numerical tolerance for verifying that the norm of the state
            vector is close to 1.

    Returns:
        A numpy ndarray corresponding to the state vector on the given
        number of
        qubits.

    Raises:
        ValueError: if num_qubits != len(qid_shape).
    """
    if isinstance(state_rep, cirq.value.ProductState):
        num_qubits = len(state_rep)

    # Check shape.
    if num_qubits is None and qid_shape is None:
        try:
            qid_shape = cirq.qis.states.infer_qid_shape(state_rep)
        except:
            raise ValueError(
                "Failed to infer the qid shape of the given state. "
                "Please specify the qid shape explicitly using either the "
                "`num_qubits` or `qid_shape` argument."
            )
    if qid_shape is None:
        qid_shape = (2,) * cast(int, num_qubits)
    else:
        qid_shape = tuple(qid_shape)
    if num_qubits is None:
        num_qubits = len(qid_shape)
    if num_qubits != len(qid_shape):
        raise ValueError(
            f"num_qubits != len(qid_shape). num_qubits is <{num_qubits!r}>. "
            f"qid_shape is <{qid_shape!r}>."
        )

    if isinstance(state_rep, np.ndarray):
        state_rep = np.copy(state_rep)
    state = cirq.quantum_state(
        state_rep, qid_shape, validate=False, dtype=dtype, atol=atol
    )
    return cast(np.ndarray, state.state_vector())


def act_on_noisy_state_vector(
    op: cirq.Operation,
    state: cirq.sim.state_vector_simulation_state.StateVectorSimulationState,
    rng: np.random.RandomState = np.random.RandomState(),
) -> None:
    # This follows eg Nielsen and Chuang eq 2.93 pg 85
    # Applies all possible Kraus operators and computes resulting probs
    # then samples a resultant state and renormalizes it
    # TODO for now only 1 qubit gates, cirq.MatrixGate limits this
    # TODO for now limit to state vector since need to calc <psi|M+M|psi>

    num_qubits = len(state.qubits)
    if not (cirq.is_measurement(op) or cirq.has_unitary(op)):
        kraus_tuple = cirq.protocols.kraus(op)
        kraus_ops = []
        for mat in kraus_tuple:
            matop = cirq.ops.matrix_gates.MatrixGate(
                mat, unitary_check=False
            ).on(op.qubits[0])
            kraus_ops.append(matop)

        probs = []
        for gate in kraus_ops:
            tmpst = state.copy()
            cirq.protocols.act_on(gate, tmpst)
            stvec = _to_unchecked_state_vector(
                tmpst.target_tensor, num_qubits=num_qubits
            )
            probs.append(np.abs(np.vdot(stvec, stvec)))
        probs = np.asarray(probs)

        # sample one of the kraus gates and apply it
        idx = rng.choice(a=len(kraus_ops), p=probs / sum(probs))
        cirq.protocols.act_on(kraus_ops[idx], state)
        # renormalize the state vector
        state._state._state_vector /= np.sqrt(probs[idx])
    else:
        cirq.protocols.act_on(op, state)
