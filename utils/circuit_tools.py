from qiskit import QuantumCircuit


def exp_P(p_string, control=None, circ=None, rot=0):
    
    num_qubits = len(p_string)
    
    #index X, Y, Z in the string of Paulis
    p_index = {}
    for index, p in enumerate(p_string):
        if p not in p_index:
            p_index[p] = [index]
        else:
            p_index[p] += [index]

    #initiate quantum circuit object
    if circ is None:
        circ = QuantumCircuit(num_qubits)
        circ.barrier()
    
    #Rotate X and Y Paulis into Z basis
    for q in p_index.keys():
        if q == 'X':
            #rotate X to Z
            for i in p_index['X']:
                circ.h(i)
            #circ.barrier()
        
        elif q == 'Y':
            #rotate Y to Z
            for i in p_index['Y']:
                circ.sdg(i)
                circ.h(i)
            #circ.barrier()
            
        else:
            pass
    
    #Evaluate parity of remaining Z qubits
    if 'I' in p_index:
        #return blank circuit if all qubits are identity
        if p_index['I'] == list(range(num_qubits)):
            return circ
        #Index qubits which are non identity
        else:
            non_I = list(set(range(num_qubits)) - set(p_index['I']))
            num_Z = range(len(non_I)-1)
    else:
        non_I = list(range(num_qubits))
        num_Z = range(num_qubits - 1)
    
    non_I = sorted(non_I)
    
    #cascade of CNOT gates between adjacent non-identity qubits
    for i in num_Z:
        circ.cx(non_I[i], non_I[i+1])
    
    #apply the rotation
    if control is None:
        circ.rz(2*rot, non_I[-1])
    else:
        circ.crz(2*rot, control, non_I[-1])
        
    #reverse cascade of CNOT gates between adjacent non-identity qubits
    for i in num_Z:
        circ.cx(non_I[len(num_Z)-i-1], non_I[len(num_Z)-i])
        
    #circ.barrier()
    
    #Rotate X and Y Paulis into Z basis
    for q in p_index.keys():
        if q == 'X':
            #rotate X to Z
            for i in p_index['X']:
                circ.h(i)
            #circ.barrier()
        
        elif q == 'Y':
            #rotate Y to Z
            for i in p_index['Y']:
                circ.h(i)
                circ.s(i)
            #circ.barrier()
            
        else:
            pass
        
    circ.barrier()
    
    return circ


def construct_ansatz(init_state=[], paulis=[], params=[], rots=[], circ=None, trot_order=1) -> QuantumCircuit:
    """
    init_state: list of qubit positions that should have value 1 (apply X). By default all 0.
    paulis: list of Pauli strings, applied left to right
    params: list of angles by which to rotate each exponentiated Pauli, leave empty to fill with parameters for optimisation in VQE
    rots: list of rotations from CS-VQE, applied left to right
    """
    if params == []:
        #parameters to be optimised in VQE routine
        param_chars = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω']
        params = [] 
        for comb in list(itertools.combinations(param_chars, 3)):
            char_str = ''.join(comb)
            params.append(Parameter(char_str))

    #initiate quantum state (usually Hartree Fock)
    if circ is None:
        circ = QuantumCircuit(len(paulis[0]))
        for q in init_state:
            circ.x(q)

    #applies the ansatz
    for t in range(trot_order):
        for index, p in enumerate(paulis):
            t_index = t*len(paulis) + index
            circ += exp_P(p_string = p, rot = params[t_index]/trot_order)
    
    #rotates in accordance with CS-VQE routine
    for r in rots:
        circ += exp_P(p_string = r[1], rot = r[0])
      
    return circ