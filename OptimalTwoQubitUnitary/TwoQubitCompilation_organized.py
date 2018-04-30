import numpy as py
import qutip as qp
from pyquil.quil import Program
import pyquil.gates as gt
import cmath, scipy, math

include_testing = True

class TwoQubitCompilation():

    def __init__(self, unitary, index=[0,1]):
        if isinstance(unitary, list):
            unitary = py.array(unitary)
        if isinstance(unitary,py.ndarray):
            assert unitary.shape == (4,4) , u"unitary must be a 4x4 matrix"
        unitary = qp.Qobj(unitary, dims=[[2,2],[2,2]])
        assert isinstance(unitary,qp.qobj.Qobj), u"unitary must be convertible to a 4x4 quantum operator."
        assert unitary.dims == [[2,2],[2,2]], u"Dimension must be [[2,2],[2,2]]."
        self.unitary = unitary
        self.qubit_index = index
        self.process()

    def process(self):
        self.decomposition_Cirac()
        self.apply_Clifford_corrections()

    def decomposition_Cirac(self):
        """
        Decompose the unitary according to the method in PRA 63, 062309.
        """
        # Step (i)
        utu = calculate_UtU(self.unitary)
        egvals, entangled_A = utu.eigenstates()
        epsilons = [0.5*(cmath.phase(x)%(2*py.pi)) for x in egvals]
        # Step (ii)
        Va, Vb, xis = finding_local_unitaries(entangled_A)
        # Step (iii)
        entangled_B = [py.exp(-1j*epsilon)*self.unitary*ket for ket,epsilon in zip(entangled_A,epsilons)]
        # Step (iv)
        Uad, Ubd, phases = finding_local_unitaries(entangled_B)
        Ua = Uad.dag()
        Ub = Ubd.dag()
        Udiag = qp.tensor([Uad,Ubd]) * self.unitary * qp.tensor([Va,Vb]).dag()

        # Correction to make (alpha,beta) positive
        two_qubit_params = list(get_alphabetagamma(Udiag))
        if two_qubit_params[0] < 0:
            two_qubit_params[0] *= -1
            two_qubit_params[1] *= -1
            Vb = qp.sigmay() * Vb
            Ub = Ub * qp.sigmay()
        if two_qubit_params[2] < 0:
            two_qubit_params[1] *= -1
            two_qubit_params[2] *= -1
            Vb = qp.sigmaz() * Vb
            Ub = Ub * qp.sigmaz()

        self.Cirac_unitaries = [tuple([Vb,Va,Ub,Ua])]
        self.Cirac_parameters = py.zeros((6,19))
        for i,matrix in enumerate(self.Cirac_unitaries[0][:4]):
            self.Cirac_parameters[0,4*i:4*i+4] = convert_unitary_to_bloch(matrix).real
        self.Cirac_parameters[0,16:19] = two_qubit_params

    def apply_Clifford_corrections(self):
        """
        Apply Clifford corrections to change the order of alpha,beta,gamma
        """
        for index in range(1,6):
            correction, operator  = clifford_operators[index]
            params_original = {x:y for x,y in zip('ZXY', self.Cirac_parameters[0,-3:])}
            two_qubit_params = [params_original[x] for x in correction]
            unitaries = list(self.Cirac_unitaries[0])
            for i in range(0,2):
                unitaries[i] = operator * unitaries[i]
            for i in range(2,4):
                unitaries[i] = unitaries[i] * operator.dag()

            if two_qubit_params[0] < 0:
                two_qubit_params[0] *= -1
                two_qubit_params[1] *= -1
                unitaries[0] = qp.sigmay() * unitaries[0]
                unitaries[2] = unitaries[2] * qp.sigmay()
            if two_qubit_params[2] < 0:
                two_qubit_params[1] *= -1
                two_qubit_params[2] *= -1
                unitaries[0] = qp.sigmaz() * unitaries[0]
                unitaries[2] = unitaries[2] * qp.sigmaz()

            self.Cirac_unitaries.append(tuple(unitaries))
            for i,matrix in enumerate(unitaries[:4]):
                self.Cirac_parameters[index,4*i:4*i+4] = convert_unitary_to_bloch(matrix).real
            self.Cirac_parameters[index,16:19] = two_qubit_params

class UsingCPhase(TwoQubitCompilation):

    def process(self):
        TwoQubitCompilation.process(self)
        self.correct = {i:{j:{} for j in range(5)} for i in range(6)}
        self.apply_correction_W0W1()
        self.apply_correction_W2W3()
        self.apply_correction_W4()

    def compile(self, pauli_index=0, clifford_index=0):
        assert isinstance(pauli_index,int), u"pauli_index must be an integer"
        assert isinstance(clifford_index,int), u"clifford_index must be an integer"
        assert clifford_index>=0 and clifford_index<6, u"clifford_index must be between 0 and 6"
        assert pauli_index>=0 and pauli_index<2**22, u"pauli_index must be between 0 and 2^22"

        bits = [(pauli_index & (1<<(n)))>>(n) for n in range(22)][::-1]
        b = bits[-10:]
        b_W0 = int(''.join([str(x) for x in bits[ 0: 3]]),base=2)
        b_W1 = int(''.join([str(x) for x in bits[ 3: 6]]),base=2)
        b_W2 = int(''.join([str(x) for x in bits[ 6: 9]]),base=2)
        b_W3 = int(''.join([str(x) for x in bits[ 9:12]]),base=2)
        b_W4 = int(''.join([str(x) for x in bits[12:20]]),base=2)

        W0 = self.correct[clifford_index][0][(
            ( b[4] + 1 ) % 2,
            ( b[0] + b[7] + b[8] ) % 2,
            ( b[8] + b[9] + 1 ) % 2
        )][b_W0]
        W1 = self.correct[clifford_index][1][(
            ( b[4] + 1 ) % 2,
            ( b[1] + b[4] + b[5] + b[7] + b[8] ) % 2,
            ( b[4] + b[8] + b[9] ) % 2
        )][b_W1]
        W2 = self.correct[clifford_index][2][(
            ( b[6] + 1 ) % 2,
            ( b[2] + b[7] + b[8] + b[9] ) % 2,
            ( b[8] ) % 2
        )][b_W2]
        W3 = self.correct[clifford_index][3][(
            ( b[6] + 1 ) % 2,
            ( b[3] + b[5] + b[6] + b[7] + b[8] + b[9] ) % 2,
            ( b[6] + b[8] + 1 ) % 2
        )][b_W3]
        W4 = self.correct[clifford_index][4][b_W4]

        return W0 + W1 + W4 + W2 + W3

    def apply_correction_W0W1(self):
        for index in range(8):
            b = [(index & (1<<(n)))>>(n) for n in range(3)][::-1]
            s = [+1 if x==0 else -1 for x in b]
            for clifford in range(6):
                gamma = self.Cirac_parameters[clifford][-1]
                for w in range(0,2):
                    unitary = rotation_z(s[0] * gamma + b[1]*py.pi) \
                            * rotation_x(s[2] * py.pi/2) \
                            * self.Cirac_unitaries[clifford][w]
                    self.correct[clifford][w][tuple(b)] = decomposition_1Q_gates(unitary, self.qubit_index[w])

    def apply_correction_W2W3(self):
        for index in range(8):
            b = [(index & (1<<(n)))>>(n) for n in range(3)][::-1]
            s = [+1 if x==0 else -1 for x in b]
            for clifford in range(6):
                beta = self.Cirac_parameters[clifford][-2]
                for w in range(2,4):
                    unitary = self.Cirac_unitaries[clifford][w] \
                            * rotation_y(s[2] * py.pi/2) \
                            * rotation_z(s[0] * beta + b[1]*py.pi)
                    self.correct[clifford][w][tuple(b)] = decomposition_1Q_gates(unitary, self.qubit_index[w%2])

    def apply_correction_W4(self):
        for index in range(2**8):
            b = [(index & (1<<(n)))>>(n) for n in range(8)][::-1]
            s = [+1 if x==0 else -1 for x in b]
            for clifford in range(6):
                alpha_corr = s[5] * self.Cirac_parameters[clifford][-3]
                beta_corr  = s[6] * self.Cirac_parameters[clifford][-2]
                gamma_corr = s[4] * self.Cirac_parameters[clifford][-1]
                program = Program()
                program.inst(gt.CPHASE(keep_within_limits(2*gamma_corr), self.qubit_index[0], self.qubit_index[1]))
                program.inst(gt.RX(+s[0]*py.pi/2, self.qubit_index[0]))
                program.inst(gt.RX(+s[1]*py.pi/2, self.qubit_index[1]))
                program.inst(gt.RZ(keep_within_limits(-alpha_corr-s[7]*s[0]*s[2]*py.pi/2), self.qubit_index[0]))
                program.inst(gt.RZ(keep_within_limits(-alpha_corr+s[7]*s[1]*s[3]*s[4]*s[5]*s[6]*py.pi/2), self.qubit_index[1]))
                program.inst(gt.CPHASE(keep_within_limits(2*alpha_corr), self.qubit_index[0], self.qubit_index[1]))
                program.inst(gt.RX(-s[2]*py.pi/2, self.qubit_index[0]))
                program.inst(gt.RX(-s[3]*py.pi/2, self.qubit_index[1]))
                program.inst(gt.CPHASE(keep_within_limits(2*beta_corr), self.qubit_index[0], self.qubit_index[1]))
                self.correct[clifford][4][index] = program

################################################################################
################################################################################
# FUNCTIONS REQUIRED FOR apply_correction
################################################################################
################################################################################

def decomposition_1Q_gates(unitary, qubit_index):
    phase = cmath.phase(scipy.linalg.det(unitary.full()))
    unitary = unitary * py.exp(-1j*phase/2.)
    u0,uz,ux,uy = convert_unitary_to_bloch(unitary)

    implementations = {}
    for index in range(8):
        b = [(index & (1<<(n)))>>(n) for n in range(3)[::-1]]
        s = [+1 if x==0 else -1 for x in b]

        if s[0]*s[1]>0:
            t0Pt2by2 = math.atan2(uz.real,u0.real)
            t0Mt2by2 = math.atan2(ux.real,uy.real)
            cost1 = py.sqrt(u0.real**2 + uz.real**2) / 2.
            sint1 = py.sqrt(ux.real**2 + uy.real**2) / 2.
        else:
            t0Pt2by2 = math.atan2(-u0.real,uz.real)
            t0Mt2by2 = math.atan2(-uy.real,ux.real)
            sint1 = py.sqrt(u0.real**2 + uz.real**2) / 2.
            cost1 = py.sqrt(ux.real**2 + uy.real**2) / 2.
        t0 = (t0Pt2by2 + t0Mt2by2)
        t2 = (t0Pt2by2 - t0Mt2by2)
        t1 = 2 * math.atan2(sint1, cost1)

        program = Program()
        program.inst(gt.RZ(keep_within_limits(t0+b[2]*py.pi), qubit_index))
        program.inst(gt.RX(+s[0]*py.pi/2, qubit_index))
        program.inst(gt.RZ(keep_within_limits(s[0]*s[2]*t1), qubit_index))
        program.inst(gt.RX(-s[1]*py.pi/2, qubit_index))
        program.inst(gt.RZ(keep_within_limits(t2+b[2]*py.pi), qubit_index))

        implementations[index] = program
    return implementations

def rotation_z(theta):
    hamiltonian = qp.sigmaz() * theta/2
    return (-1j*hamiltonian).expm()
def rotation_x(theta):
    hamiltonian = qp.sigmax() * theta/2
    return (-1j*hamiltonian).expm()
def rotation_y(theta):
    hamiltonian = qp.sigmay() * theta/2
    return (-1j*hamiltonian).expm()
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


################################################################################
# END OF
# FUNCTIONS REQUIRED FOR apply_correction
################################################################################


################################################################################
################################################################################
# FUNCTIONS REQUIRED FOR apply_Clifford_corrections
################################################################################
################################################################################

clifford_operators = {
    0: ('ZXY', qp.qeye(2)),
    1: ('ZYX', (qp.sigmax() + qp.sigmay()) / py.sqrt(2)),
    2: ('XZY', (qp.sigmaz() + qp.sigmax()) / py.sqrt(2)),
    3: ('XYZ', (-1j*qp.qeye(2) - qp.sigmaz() - qp.sigmax() + qp.sigmay())/2),
    4: ('YXZ', (qp.sigmay() + qp.sigmaz()) / py.sqrt(2)),
    5: ('YZX', (-1j*qp.qeye(2) - qp.sigmaz() - qp.sigmax() - qp.sigmay())/2),
}

################################################################################
# END OF
# FUNCTIONS REQUIRED FOR apply_Clifford_corrections
################################################################################

################################################################################
################################################################################
# FUNCTIONS REQUIRED FOR decomposition_Cirac
################################################################################
################################################################################

def convert_unitary_to_bloch(unitary):
    phase = cmath.phase(scipy.linalg.det(unitary.full()))
    unitary *= py.exp(-1j*phase/2.)
    u0 = ((unitary).tr() / 2.0)
    u1 = ((unitary*qp.sigmaz()).tr() * 1j / 2.0)
    u2 = ((unitary*qp.sigmax()).tr() * 1j / 2.0)
    u3 = ((unitary*qp.sigmay()).tr() * 1j / 2.0)
    return py.array([u0,u1,u2,u3])

def convert_bloch_to_unitary(bloch):
    basis = [qp.qeye(2),-1j*qp.sigmaz(),-1j*qp.sigmax(),-1j*qp.sigmay()]
    return sum([x*y for x,y in zip(bloch,basis)])

def calculate_UtU(operator):
    operator_magical = convert_to_magicalbasis(operator)
    transpose = operator_magical.full().T
    transpose = qp.Qobj(transpose,dims=operator.dims)
    utu = (transpose * operator_magical)
    return convert_from_magicalbasis(utu)

def finding_local_unitaries(entangled_basis):
    # Convert the basis vectors so that they are real in the magical basis
    real_vectors, phases = [],[]
    for vector in entangled_basis:
        real_vector,phase = make_vector_real(convert_to_magicalbasis(vector))
        real_vectors.append(real_vector)
        phases.append(phase)

    ket_ef = convert_from_magicalbasis(real_vectors[0] + 1j * real_vectors[1]) / py.sqrt(2)
    ket_epfp = convert_from_magicalbasis(real_vectors[0] - 1j * real_vectors[1]) / py.sqrt(2)

    ket_e = qp.ptrace(ket_ef,[0]).eigenstates()[1][-1]
    ket_f = qp.ptrace(ket_ef,[1]).eigenstates()[1][-1]
    ket_ep = qp.ptrace(ket_epfp,[0]).eigenstates()[1][-1]
    ket_fp = qp.ptrace(ket_epfp,[1]).eigenstates()[1][-1]
    # Must correct phases
    identity = qp.tensor([qp.qeye(2)]*2)
    phase_correction = identity.matrix_element(qp.tensor(ket_e,ket_f).dag(),ket_ef)
    ket_f = phase_correction * ket_f
    phase_correction = identity.matrix_element(qp.tensor(ket_ep,ket_fp).dag(),ket_epfp)
    ket_fp = phase_correction * ket_fp

    # Now I must rewrite the Psi_bar states in this basis
    local_basis = [qp.tensor(x,y) for x in [ket_e,ket_ep] for y in [ket_f,ket_fp]]
    matrix_t = []
    for base in real_vectors:
        base_comp = convert_from_magicalbasis(base)
        matrix_t.append([identity.matrix_element(x.dag(),base_comp) for x in local_basis])
    matrix = py.array(matrix_t).T

    # Find phase 'delta'
    delta = (py.pi/2 + cmath.phase(matrix[1,2]))%(2*py.pi)

    unitary_A = (qp.basis(2,0) * ket_e.dag() + qp.basis(2,1) * ket_ep.dag() * py.exp(+1j*delta))
    unitary_B = (qp.basis(2,0) * ket_f.dag() + qp.basis(2,1) * ket_fp.dag() * py.exp(-1j*delta))

    # Finding the phases to get magical basis
    # For some reason, the order of the magical basis must be changed
    phases = []
    transform = qp.tensor([unitary_A,unitary_B])
    changed_order = [magical_vectors[x] for x in [0,1,3,2]]
    for vector,magical in zip(entangled_basis,changed_order):
        magical_ket = qp.Qobj(py.array(magical).T,dims=[[2,2],[1,1]]) / py.sqrt(2)
        vec = transform * vector
        inner_product = identity.matrix_element(magical_ket.dag(),vec)
        phase = - cmath.phase(inner_product)
        phases.append( phase%(2*py.pi) )

    return unitary_A,unitary_B,phases

def make_vector_real(vector):
    phases = [cmath.phase(x)%py.pi for x in vector]
    diff = py.diff(phases + phases[0:1])
    assert min(abs(diff)) < 1e-2 , u"Vector can not be made real"
    phase = phases[0]
    vector = vector * py.exp(-1j*phase)
    return vector,phase

magical_vectors = []
magical_vectors.append([1,0,0,1])
magical_vectors.append([-1j,0,0,1j])
magical_vectors.append([0,1,-1,0])
magical_vectors.append([0,-1j,-1j,0])
magical_basis = qp.Qobj(py.array(magical_vectors).T/py.sqrt(2),dims=[[2,2],[2,2]])
magical_basis_dag = magical_basis.dag()

def convert_to_magicalbasis(operator):
    if operator.type == 'oper':
        return magical_basis_dag * operator * magical_basis
    if operator.type == 'ket':
        return magical_basis_dag * operator
    if operator.type == 'bra':
        return operator * magical_basis

def convert_from_magicalbasis(operator):
    if operator.type == 'oper':
        return magical_basis * operator * magical_basis_dag
    if operator.type == 'ket':
        return magical_basis * operator
    if operator.type == 'bra':
        return operator * magical_basis

def get_alphabetagamma(unitary):
    eigenvalues, eigenstates = unitary.eigenstates()
    phases = [cmath.phase(x) for x in eigenvalues]
    hamiltonian = -sum([x*qp.ket2dm(y) for x,y in zip(phases,eigenstates)])
    hamiltonian = qp.Qobj(hamiltonian,dims = [[2,2],[2,2]])
    alpha = qp.expect(hamiltonian, qp.tensor([qp.sigmaz()]*2)) / 2.0
    beta = qp.expect(hamiltonian, qp.tensor([qp.sigmax()]*2)) / 2.0
    gamma = qp.expect(hamiltonian, qp.tensor([qp.sigmay()]*2)) / 2.0
    return alpha, beta, gamma

################################################################################
# END OF
# FUNCTIONS REQUIRED FOR decomposition_Cirac
################################################################################

################################################################################
################################################################################
# TESTING
################################################################################
################################################################################

if include_testing:

    import unittest, random
    import pyquil.api as api
    qvm = api.QVMConnection()

    class test_TwoQubitCompilation(unittest.TestCase):

        def QVM_get_unitary_matrix(self, program, indices = [0,1]):
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

        def unitary_equivalence(self, uA, uB):
            thisCorrect = True
            should_be_identity = uA.dag() * uB
            eigs = should_be_identity.eigenenergies() * (1+0j)
            eigs*= py.exp(-1j*cmath.phase(eigs[0]))
            for eig in eigs:
                if abs(eig-1) > 1e-6:
                    thisCorrect = False
                    break
            return thisCorrect

        def test_compile(self):
            unitary = qp.rand_unitary(4)
            compiled = UsingCPhase(unitary)
            successes, failures = 0, 0
            try:
                while True:
                    pauli = random.randrange(2**22)
                    clifford = random.randrange(6)
                    circuit = compiled.compile(pauli, clifford)
                    reconstruct = self.QVM_get_unitary_matrix(circuit,compiled.qubit_index)
                    thisCorrect = self.unitary_equivalence(qp.Qobj(reconstruct),unitary)
                    if thisCorrect:
                        successes += 1
                    else:
                        failures += 1
                    print('So far, {} successes and {} failures.'.format(successes,failures))
                    print('\tTested clifford = {} and pauli = {}.'.format(clifford, pauli))
                    with self.subTest(pauli = pauli, clifford = clifford):
                        self.assertTrue(thisCorrect)
            except KeyboardInterrupt:
                pass

################################################################################
# END OF
# FUNCTIONS REQUIRED FOR decomposition_Cirac
################################################################################
