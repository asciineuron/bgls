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

"""BGLS is a Cirq implementation of Bravyi, Gosset, and Liu's algorithm for
simulating quantum measurement without computing marginals
(https://arxiv.org/abs/2112.08499).
"""

from bgls.simulator import Simulator
from bgls.utils import (
    cirq_state_vector_bitstring_probability,
    cirq_density_matrix_bitstring_probability,
    cirq_stabilizer_ch_bitstring_probability,
    apply_near_clifford_gate,
)
