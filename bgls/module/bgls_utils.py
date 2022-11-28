# helper functions to operate on cirq structures

import cirq
import numpy as np
from typing import TypeVar

State = TypeVar("State")


def state_vector_bitstring_amplitude(
    state_vector: cirq.sim.state_vector_simulation_state.StateVectorSimulationState,
    bitstring: str,
) -> complex:
    """
    For a cirq StateVectorSimulationState,
    returns the amplitude corresponding to a given bitstring
    """
    index = int(bitstring, 2)
    vector = cirq.to_valid_state_vector(state_vector.target_tensor)
    amplitude = np.abs(vector[index])
    return amplitude


def density_matrix_bitstring_probability(
    state: cirq.sim.DensityMatrixSimulationState, bitstring: str
) -> float:
    index = int(bitstring, 2)
    density_matrix = cirq.to_valid_density_matrix(state.target_tensor)
    probability = 0  # tr (prod_m (sum_j p_j |psi_j><psi_j|))
    return probability
