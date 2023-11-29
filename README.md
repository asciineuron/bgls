# BGLS

[![build](https://github.com/asciineuron/bgls/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/asciineuron/bgls/actions)
[![arXiv](https://img.shields.io/badge/arXiv-2311.11787-<COLOR>.svg)](https://arxiv.org/abs/2311.11787)
[![doctest](https://github.com/asciineuron/bgls/actions/workflows/doctest.yml/badge.svg?branch=main)](https://github.com/asciineuron/bgls/actions)
[![pages-build-deployment](https://github.com/asciineuron/bgls/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/asciineuron/bgls/actions/workflows/pages/pages-build-deployment)
[![Documentation](https://img.shields.io/badge/Documentation-GH_Pages-blue)](https://asciineuron.github.io/bgls/)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/asciineuron/bgls)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

BGLS is a Python package that implements the **B**ravyi, **G**osset, and **L**iu **S**ampling algorithm presented in _How to simulate quantum measurement without computing marginals ([Phys. Rev. Lett.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.220503)) ([arXiv](https://arxiv.org/abs/2112.08499))_ for [Cirq](https://quantumai.google/cirq) circuits.

## Quickstart

### Installation

```bash
pip install bgls
```

### Example

```python
import cirq
import bgls

# Example circuit to run.
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H.on(qubits[0]),
    cirq.CNOT.on(*qubits),
    cirq.measure(*qubits, key="z")
)

# Run the circuit with BGLS.
simulator = bgls.Simulator(
    initial_state=cirq.StateVectorSimulationState(qubits=qubits, initial_state=0),
    apply_op=cirq.protocols.act_on,
    compute_probability=bgls.born.compute_probability_state_vector,
)
results = simulator.run(circuit, repetitions=10)
print(results.histogram(key="z"))
```

Sample output:

```text
Counter({0: 6, 3: 4})
```

## Documentation

See more details and examples in the [Documentation for BGLS](https://asciineuron.github.io/bgls/).
