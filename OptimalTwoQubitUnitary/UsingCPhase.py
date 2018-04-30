import pylab as py
from pyquil.quil import Program
import pyquil.api as api
import pyquil.gates as gt
qvm = api.QVMConnection()
import itertools
import cmath

try:
    import qutip as qp
except ModuleNotFoundError:
    pass

def rotation_z(theta):
    hamiltonian = qp.sigmaz() * theta/2
    return (-1j*hamiltonian).expm()
def rotation_x(theta):
    hamiltonian = qp.sigmax() * theta/2
    return (-1j*hamiltonian).expm()
def rotation_y(theta):
    hamiltonian = qp.sigmay() * theta/2
    return (-1j*hamiltonian).expm()

def two_qubit_unitary(alpha, beta, gamma, index=[0,1]):
    """
    Implements the two-qubit unitary
        U = e^{-iH}
    where
        H = (alpha/2) * ZZ + (beta/2) * XX + (gamma/2) * YY
    in the form
        U = (W3 x W2) W4 (W1 x W0)
    """
    implementations = []
    for kx in itertools.product([+1,-1],repeat=2):
        for ky in itertools.product([+1,-1],repeat=2):
            for kz in itertools.product([+1,-1],repeat=2):

                gamma_corr = ky[0]*ky[1]*gamma
                alpha_corr = alpha
                beta_corr = kx[0]*kx[1]*kz[0]*kz[1]*ky[0]*ky[1]*beta
                angle_x1 = lambda s: beta_corr + py.pi/2 if s>0 else beta_corr - py.pi/2
                angle_z1 = lambda s: alpha_corr - py.pi/2 if s>0 else alpha_corr + py.pi/2

                W0 = rotation_z(gamma_corr) * rotation_x(-ky[0]*py.pi/2)
                W1 = rotation_z(gamma_corr) * rotation_x(-ky[1]*py.pi/2)
                W2 = rotation_y(+kx[0]*kz[0]*ky[0]*py.pi/2) * rotation_z(angle_x1(kz[0]))
                W3 = rotation_y(+kx[1]*kz[1]*ky[1]*py.pi/2) * rotation_z(angle_x1(kz[1]))

                program = Program()
                program.inst(gt.CPHASE(keep_within_limits(-2*gamma_corr), index[0], index[1]))
                for i in range(2):
                    program.inst(gt.RX(+ky[i]*py.pi/2, index[i]))
                for i in range(2):
                    program.inst(gt.RZ(keep_within_limits(angle_z1(kz[i])), index[i]))
                program.inst(gt.CPHASE(keep_within_limits(-2*alpha_corr), index[0], index[1]))
                for i in range(2):
                    program.inst(gt.RX(-kx[i]*ky[i]*py.pi/2, index[i]))
                program.inst(gt.CPHASE(keep_within_limits(-2*beta_corr), index[0], index[1]))

                yield (W0,W1,W2,W3,program)
                # implementations.append((W0,W1,W2,W3,program))
    # return implementations

def apply_pauli_corrections(alpha, beta, gamma, index=[0,1]):
    paulis = {'I': qp.qeye(2), 'Z': qp.sigmaz(), 'X': qp.sigmax(), 'Y': qp.sigmay()}
    for corrections in itertools.product('IZXY',repeat=2):
        params = [alpha, beta, gamma]
        for i in range(2):
            if corrections[i] == 'Z':
                params = [x*y for x,y in zip(params,[+1,-1,-1])]
            elif corrections[i] == 'X':
                params = [x*y for x,y in zip(params,[-1,+1,-1])]
            elif corrections[i] == 'Y':
                params = [x*y for x,y in zip(params,[-1,-1,+1])]
        for W0,W1,W2,W3,program in two_qubit_unitary(*params, index):
            new_W0 = W0 * paulis[corrections[0]]
            new_W1 = W1 * paulis[corrections[1]]
            new_W2 = paulis[corrections[0]] * W2
            new_W3 = paulis[corrections[1]] * W3
            yield (new_W0,new_W1,new_W2,new_W3,program)

def apply_clifford_corrections(alpha, beta, gamma, index=[0,1]):
    cliffords = {
        'ZXY': qp.qeye(2),
        'XZY': (qp.sigmaz() + qp.sigmax()) / py.sqrt(2),
        'ZYX': (qp.sigmax() + qp.sigmay()) / py.sqrt(2),
        'YXZ': (qp.sigmay() + qp.sigmaz()) / py.sqrt(2),
        'XYZ': (-1j*qp.qeye(2) - qp.sigmaz() - qp.sigmax() + qp.sigmay())/2,
        'YZX': (-1j*qp.qeye(2) - qp.sigmaz() - qp.sigmax() - qp.sigmay())/2,
    }
    params_original = {'Z': alpha, 'X': beta, 'Y': gamma}
    for correction, operator in cliffords.items():
        params = [params_original[x] for x in correction]

        # pauli_corrections = list(apply_pauli_corrections(*params, index))
        # for W0,W1,W2,W3,program in pauli_corrections[:1]:
        for W0,W1,W2,W3,program in apply_pauli_corrections(*params, index):
            new_W0 = W0 * operator
            new_W1 = W1 * operator
            new_W2 = operator.dag() * W2
            new_W3 = operator.dag() * W3
            yield (new_W0,new_W1,new_W2,new_W3,program)

def QVM_get_unitary_matrix(program, indices):
    dims = 2**len(indices)
    unitary = py.zeros((dims,dims)) * 0j
    for i in range(dims):
        preparation = Program()
        binary = bin(i)[2:].zfill(len(indices))[::-1]
        for c in range(len(indices)):
            if binary[c]=='1':
                preparation.inst(gt.X(indices[c]))
        wavefunction = qvm.wavefunction(preparation + program)
        unitary[:,i] = wavefunction.amplitudes
    return unitary

def unitary_equivalence(uA, uB):
    thisCorrect = True
    should_be_identity = uA.dag() * uB
    eigs = should_be_identity.eigenenergies() * (1+0j)
    eigs*= py.exp(-1j*cmath.phase(eigs[0]))
    for eig in eigs:
        if abs(eig-1) > 1e-6:
            thisCorrect = False
            break
    return thisCorrect

def check_equivalence(pauliA, pauliB):
    for i in range(4):
        if not unitary_equivalence(pauliA[i], pauliB[i]):
            return False
    return True

"""
Algorithm to categorize elements into equivalence classes
http://code.activestate.com/recipes/499354-equivalence-partition/
"""
def equivalence_partition( iterable, relation ):
    """
    Partitions a set of objects into equivalence classes,
    Args:
        iterable: collection of objects to be partitioned
        relation: equivalence relation. I.e. relation(o1,o2) evaluates to True
            if and only if o1 and o2 are equivalent
    Returns: classes, partitions
        partitions: A dictionary mapping objects to equivalence classes
    """
    class_representatives = []
    partitions = {}
    try:
        for counter, obj in enumerate(iterable): # for each object
        # find the class it is in\n",
            found = False
            for k, element in enumerate(class_representatives):
                if relation( element, obj ): # is it equivalent to this class?
                    partitions[counter] = k
                    found = True
                    break
            if not found: # it is in a new class\n",
                class_representatives.append( obj )
                partitions[counter] = len(class_representatives) - 1
            print('Tested {:d} elements and found {:d} classes'.format(counter+1,len(class_representatives)))
    except KeyboardInterrupt:
        pass
    return class_representatives, partitions

def keep_within_limits(theta, limits=None):
    """
    Keep an angle between +pi and -pi
    """
    if limits==None:
        limits = [-py.pi,+py.pi]
    if limits[1] < limits[0]:
        limits = limits[0:2][::-1]
    interval = limits[1] - limits[0]
    while (theta <= limits[0]):
        theta += interval
    while (theta > limits[1]):
        theta -= interval
    return theta

def apply_pauli_corrections_V2(alpha, beta, gamma, implementation_index = None, qubit_index=[0,1]):
    if implementation_index==None:
        implementations = []
        for i in range(2**10):
            implementations.append(apply_pauli_corrections_V2(alpha, beta, gamma, i, qubit_index))
        return implementations
    else:

        binary = bin(implementation_index)[2:].zfill(10)
        p0 = int(binary[0:2],base=2)
        p1 = int(binary[2:4],base=2)
        kx = [+1 if x=='0' else -1 for x in binary[4:6]]
        ky = [+1 if x=='0' else -1 for x in binary[6:8]]
        kz = [+1 if x=='0' else -1 for x in binary[8:10]]
        # select_bit = lambda x,n : (x & (1<<n))>>n
        # select_base4 = lambda x,n : (x & (3<<(2*n)))>>(2*n)
        # select_base8 = lambda x,n : (x & (7<<(3*n)))>>(3*n)
        # select_base16 = lambda x,n : (x & (15<<(4*n)))>>(4*n)

        paulis = {0: qp.qeye(2), 1: qp.sigmaz(), 2: qp.sigmax(), 3: qp.sigmay()}

        params = [alpha, beta, gamma]

        for i in [p0,p1]:
            if i == 1:
                params = [x*y for x,y in zip(params,[+1,-1,-1])]
            elif i == 2:
                params = [x*y for x,y in zip(params,[-1,+1,-1])]
            elif i == 3:
                params = [x*y for x,y in zip(params,[-1,-1,+1])]


        gamma_corr = ky[0] * ky[1] * params[2]
        alpha_corr = params[0]
        beta_corr = kx[0] * kx[1] * kz[0] * kz[1] * ky[0] * ky[1] * params[1]

        angle_x1 = lambda s: beta_corr + py.pi/2 if s>0 else beta_corr - py.pi/2
        angle_z1 = lambda s: alpha_corr - py.pi/2 if s>0 else alpha_corr + py.pi/2

        W0 = rotation_z(gamma_corr) * rotation_x(-ky[0]*py.pi/2) * paulis[p0]
        W1 = rotation_z(gamma_corr) * rotation_x(-ky[1]*py.pi/2) * paulis[p1]
        W2 = paulis[p0] * rotation_y(+kx[0]*kz[0]*ky[0]*py.pi/2) * rotation_z(angle_x1(kz[0]))
        W3 = paulis[p1] * rotation_y(+kx[1]*kz[1]*ky[1]*py.pi/2) * rotation_z(angle_x1(kz[1]))

        program = Program()
        program.inst(gt.CPHASE(keep_within_limits(-2*gamma_corr), qubit_index[0], qubit_index[1]))
        for i in range(2):
            program.inst(gt.RX(+ky[i]*py.pi/2, qubit_index[i]))
        for i in range(2):
            program.inst(gt.RZ(keep_within_limits(angle_z1(kz[i])), qubit_index[i]))
        program.inst(gt.CPHASE(keep_within_limits(-2*alpha_corr), qubit_index[0], qubit_index[1]))
        for i in range(2):
            program.inst(gt.RX(-kx[i]*ky[i]*py.pi/2, qubit_index[i]))
        program.inst(gt.CPHASE(keep_within_limits(-2*beta_corr), qubit_index[0], qubit_index[1]))

        return  (W0, W1, W2, W3, program)
