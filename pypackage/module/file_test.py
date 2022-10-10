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

"""Unit tests for file.py."""

import pytest

import numpy as np
from pypackage.module.file import Object


@pytest.mark.parametrize("items", [[1, 2, 10], [np.pi, np.e]])
def test_object(items):
    obj = Object(items)
    assert obj.num_items == len(items)
    assert np.allclose(obj.asarray, np.array(items))
