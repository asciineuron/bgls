from typing import List, Tuple

import numpy as np

import cirq


def is_clifford_gate(op: cirq.Operation) -> bool:
    # could do cirq.has_stabilizer_effect(op), not sure if covers other
    # gates though
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
    # yields a list of clifford circuits approximating the near-clifford input
    # for now only expands T non-clifford gates.
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
