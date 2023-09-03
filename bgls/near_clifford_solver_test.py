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

"""Tests for the BGLS Simulator near clifford solver."""

import pytest
import numpy as np

import cirq

import bgls
import bgls.near_clifford_solver
import bgls.testing


def expansions_equal(exp1, exp2):
    # returns true if each list of circuits is the same (cannot convert to set)
    # to check more easily
    for circuit in exp1:
        if circuit in exp2:
            exp2.remove(circuit)
        else:
            return False
    return True


def test_expanded_clifford():
    """Each circuit in circuit_clifford_decomposition should be pure
    clifford"""
    qubits = cirq.LineQubit.range(3)
    domain = {cirq.H, cirq.CNOT, cirq.S, cirq.T}
    clifford_circuit = bgls.testing.generate_random_circuit(
        qubits,
        n_moments=10,
        op_density=0.5,
        gate_domain=domain,
        random_state=1,
    )
    (
        expanded_circuits,
        expanded_amplitudes,
    ) = bgls.near_clifford_solver.circuit_clifford_decomposition(
        clifford_circuit
    )

    for circuit in expanded_circuits:
        for op in circuit.all_operations():
            assert cirq.has_stabilizer_effect(op)


def test_pure_clifford():
    """If no non-clifford gates returns original circuit"""
    qubits = cirq.LineQubit.range(3)
    domain = {cirq.H, cirq.CNOT, cirq.S}
    clifford_circuit = bgls.testing.generate_random_circuit(
        qubits,
        n_moments=10,
        op_density=0.5,
        gate_domain=domain,
        random_state=1,
    )
    (
        expanded_circuits,
        expanded_amplitudes,
    ) = bgls.near_clifford_solver.circuit_clifford_decomposition(
        clifford_circuit
    )
    assert (
        len(expanded_circuits) == 1
        and expanded_circuits[0] == clifford_circuit
    )
    assert len(expanded_amplitudes) == 1 and expanded_amplitudes[0] == 1.0


def test_clifford_expansion():
    """A circuit with one T will expand into two circuits,
    one with I, one with S"""
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H.on(q[0]), cirq.H.on(q[1]), cirq.H.on(q[2]), cirq.T.on(q[1])
    )
    (
        expanded_circuits,
        expanded_amplitudes,
    ) = bgls.near_clifford_solver.circuit_clifford_decomposition(circuit)

    test_exp = [
        cirq.Circuit(
            cirq.H.on(q[0]), cirq.H.on(q[1]), cirq.H.on(q[2]), cirq.I.on(q[1])
        ),
        cirq.Circuit(
            cirq.H.on(q[0]), cirq.H.on(q[1]), cirq.H.on(q[2]), cirq.S.on(q[1])
        ),
    ]

    assert expansions_equal(expanded_circuits, test_exp)


def test_larger_clifford_expansion():
    """The number of circuits generated is fidelity * 2**num T gates"""
    qubits = cirq.LineQubit.range(3)
    domain = {cirq.H, cirq.CNOT, cirq.S, cirq.T}
    clifford_circuit = bgls.testing.generate_random_circuit(
        qubits,
        n_moments=50,
        op_density=0.5,
        gate_domain=domain,
        random_state=1,
    )
    num_t = 0
    for op in clifford_circuit.all_operations():
        if (
            isinstance(op.gate, cirq.ops.common_gates.ZPowGate)
            and op.gate.exponent == 0.25
        ):
            num_t += 1
    fidelity = 0.6
    (
        expanded_circuits,
        expanded_amplitudes,
    ) = bgls.near_clifford_solver.circuit_clifford_decomposition(
        clifford_circuit, fidelity=fidelity
    )

    assert len(expanded_circuits) == int(fidelity * 2**num_t) and len(
        expanded_amplitudes
    ) == int(fidelity * 2**num_t)


@pytest.mark.parametrize("rad", [0.1, 0.25, 1.0, np.pi + 1e-5])
def test_rz_circuit_expansion(rad: float):
    """The decomposition works on generic Rz gates as well."""
    qubits = cirq.LineQubit.range(3)
    domain = {cirq.H, cirq.CNOT, cirq.S, cirq.Rz(rads=rad)}
    clifford_circuit = bgls.testing.generate_random_circuit(
        qubits,
        n_moments=50,
        op_density=0.5,
        gate_domain=domain,
        random_state=1,
    )
    num_rz = 0
    for op in clifford_circuit.all_operations():
        if not cirq.has_stabilizer_effect(op) and (
            isinstance(op.gate, cirq.ops.common_gates.ZPowGate)
        ):
            num_rz += 1
    fidelity = 0.6
    (
        expanded_circuits,
        expanded_amplitudes,
    ) = bgls.near_clifford_solver.circuit_clifford_decomposition(
        clifford_circuit, fidelity=fidelity
    )

    assert len(expanded_circuits) == int(fidelity * 2**num_rz) and len(
        expanded_amplitudes
    ) == int(fidelity * 2**num_rz)
