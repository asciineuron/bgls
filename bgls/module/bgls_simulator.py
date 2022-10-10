import cirq
import numpy as np
from typing import Dict

class GateSamplerSimulator(cirq.SimulatesSamples):
    def __init__(self):
        # TODO
        print("add features as necessary")
    
    def _run(self, circuit: cirq.Circuit, param_resolver: cirq.ParamResolver, repetitions: int) -> Dict[str, np.ndarray]:
        # from qsim, use to resolve parameters if passed
        param_resolver = param_resolver or cirq.ParamResolver({})
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        
        return self._sample_final_results(resolved_circuit, repetitions)

    def _sample_final_results(self, circuit: cirq.Circuit, repetitions: int) -> Dict[str, np.ndarray]:
        # TODO
        return {}

    # that's it to implement the sampler simulator interface