import os
from qiskit.algorithms import VQE
from qiskit import IBMQ
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo, FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, TensoredMeasFitter
import utils.qonversion_tools as qonvert


def get_quantum_instance(seed, noise=False, run_local=True, error_mitigation=None):
    algorithm_globals.random_seed = seed
    
    if not run_local:
        with open('data/token.txt', 'r') as file:
            TOKEN = file.read()
        IBMQ.save_account(TOKEN, overwrite=True)
        provider = IBMQ.get_provider(hub='ibm-q')
            
    if not noise:
        if run_local:
            backend = Aer.get_backend('statevector_simulator')
            qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
            
        
    else:
        device_backend = FakeRueschlikon()
        backend = Aer.get_backend('aer_simulator')
        noise_model = None
        device = QasmSimulator.from_backend(device_backend)
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates
        
        qi = QuantumInstance(backend=backend,
                             shots=2**10,
                             seed_simulator=seed, 
                             seed_transpiler=seed,
                             coupling_map=coupling_map, 
                             noise_model=noise_model,
                             measurement_error_mitigation_cls=error_mitigation,
                             measurement_error_mitigation_shots=2**10,
                             cals_matrix_refresh_period=30)
        
    return qi


def vqe_simulation(ansatz, operator, init_params, noise=False, error_mitigation=None):
    
    def store_intermediate_result(eval_count, parameters, mean, std):
        """ Outputs intermediate data during VQE routine
        """
        counts.append(eval_count)
        prmset.append(parameters)
        values.append(mean)
        errors.append(std)
    
    counts=[]
    prmset=[]
    values=[]
    errors=[]
    
    qi = get_quantum_instance(seed=42, noise=noise, error_mitigation=error_mitigation)
    vqe = VQE(ansatz, 
              initial_point=init_params, 
              optimizer=IMFIL(maxiter=1000), 
              callback=store_intermediate_result, 
              quantum_instance=qi) 
    vqe_input_op = qonvert.dict_to_WeightedPauliOperator(operator)
    vqe_run = vqe.compute_minimum_eigenvalue(operator=vqe_input_op)
    
    return {'conval':vqe_run.optimal_value,
            'counts':counts,
            'prmset':prmset,
            'values':values,
            'errors':errors}


def remote_VQE(operator, qc, init_params):
    
    vqe_input_op = qonvert.dict_to_WeightedPauliOperator(operator)
    
    optimizer = {
        'name': 'QN-SPSA',  # leverage the Quantum Natural SPSA
        # 'name': 'SPSA',  # set to ordinary SPSA
        'maxiter': 200,
        'resamplings': {1: 200},  # 100 samples of the QFI for the first step, then 1 sample per step
    }

    provider = IBMQ.get_provider(
        hub='ibm-q',
        group='open',
        project='main'
    )

    options = {
        'backend_name': 'ibmq_qasm_simulator'
    }

    runtime_inputs = {
      'ansatz': qc,
      'operator': vqe_input_op,
      'optimizer': optimizer,
      'initial_parameters': init_params,
      'measurement_error_mitigation': True
    }

    job = provider.runtime.run(
        program_id='vqe',
        options=options,
        inputs=runtime_inputs
    )
    
    return job.result()