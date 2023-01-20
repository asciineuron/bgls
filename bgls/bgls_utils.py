# helper functions to operate on cirq structures

import cirq
import numpy as np
from typing import TypeVar

State = TypeVar("State")


def compute_state_vector_probability(
    state_vector: cirq.sim.state_vector_simulation_state.StateVectorSimulationState,
    bitstring: str,
) -> float:
    """
    For a cirq StateVectorSimulationState,
    returns the probability corresponding to a given bitstring
    """
    return (
        np.abs(
            cirq.to_valid_state_vector(state_vector.target_tensor)[
                int(bitstring, 2)
            ]
        )
        ** 2
    )


def density_matrix_bitstring_probability(
    state: cirq.sim.DensityMatrixSimulationState, bitstring: str
) -> float:
    """
    For a cirq DensityMatrixSimulationState,
    returns the probability corresponding to a given bitstring
    """
    num_qubits = len(bitstring)
    index = int(bitstring, 2)
    density_matrix = cirq.to_valid_density_matrix(
        state.target_tensor, num_qubits
    )
    return np.abs(density_matrix[index, index])
