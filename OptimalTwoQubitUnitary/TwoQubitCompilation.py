import pylab as py
from pyquil.quil import Program
import pyquil.api as api
import pyquil.gates as gt
qvm = api.QVMConnection()
import cmath
import scipy.linalg, math, itertools

import qutip as qp

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
        self.get_alphabetagamma()

    def compile(self, pauli = None, clifford = None):
        if clifford is None:
            clifford = range(6)
        if isinstance(clifford, int):
            clifford = [clifford]
        if pauli is None:
            pauli = range(2**(10+4*3))
        if isinstance(pauli, int):
            pauli = [pauli]
        out = {}
        for cl in clifford:
            self.apply_Clifford_correction(cl)
            for pl in pauli:

                rest, two_qubit_index = divmod(pl, 2**10)
                W_index = []
                for i in range(4):
                    rest, ind = divmod(rest, 8)
                    W_index.append(ind)

                self.gen_twoqubitcircuit(two_qubit_index)

                program = decomposition_1Q_gates(self.pauli_W0, W_index[0], self.qubit_index[0])
                program+= decomposition_1Q_gates(self.pauli_W1, W_index[1], self.qubit_index[1])
                program+= self.pauli_program
                program+= decomposition_1Q_gates(self.pauli_W2, W_index[2], self.qubit_index[0])
                program+= decomposition_1Q_gates(self.pauli_W3, W_index[3], self.qubit_index[1])
                out[(pl,cl)] = program

                # out[(cl,pl)] = tuple([
                #     self.pauli_W0,
                #     self.pauli_W1,
                #     self.pauli_W2,
                #     self.pauli_W3,
                #     self.pauli_program,
                # ])
                # print('Done i = {} and j = {}'.format(cl,pl))

        return out

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
        self.standard_W0 = Vb
        self.standard_W1 = Va
        self.standard_W2 = Ub
        self.standard_W3 = Ua
        self.standard_W4 = Udiag
        return Va, Vb, Ua, Ub, Udiag

    def get_alphabetagamma(self):
        eigenvalues, eigenstates = self.standard_W4.eigenstates()
        phases = [cmath.phase(x) for x in eigenvalues]
        hamiltonian = -sum([x*qp.ket2dm(y) for x,y in zip(phases,eigenstates)])
        hamiltonian = qp.Qobj(hamiltonian,dims = [[2,2],[2,2]])
        alpha = qp.expect(hamiltonian, qp.tensor([qp.sigmaz()]*2)) / 2.0
        beta = qp.expect(hamiltonian, qp.tensor([qp.sigmax()]*2)) / 2.0
        gamma = qp.expect(hamiltonian, qp.tensor([qp.sigmay()]*2)) / 2.0
        self.standard_params = (alpha, beta, gamma)
        return alpha, beta, gamma


def decomposition_1Q_gates(unitary, index = 0, qubit_index=0):
    phase = cmath.phase(scipy.linalg.det(unitary.full()))
    unitary = unitary * py.exp(-1j*phase/2.)
    u0,u1,u2,u3 = convert_unitary_to_bloch(unitary)

    b = [(index & (1<<(n)))>>(n) for n in range(3)[::-1]]
    s = [+1 if x==0 else -1 for x in b]

    if s[0]*s[1]>0:
        t0Pt2by2 = math.atan2(u3.real,u0.real)
        t0Mt2by2 = math.atan2(u1.real,u2.real)
        cost1 = py.sqrt(u0.real**2 + u3.real**2) / 2.
        sint1 = py.sqrt(u1.real**2 + u2.real**2) / 2.
    else:
        t0Pt2by2 = math.atan2(-u0.real,u3.real)
        t0Mt2by2 = math.atan2(-u2.real,u1.real)
        sint1 = py.sqrt(u0.real**2 + u3.real**2) / 2.
        cost1 = py.sqrt(u1.real**2 + u2.real**2) / 2.
    t0 = (t0Pt2by2 + t0Mt2by2)
    t2 = (t0Pt2by2 - t0Mt2by2)
    t1 = 2 * math.atan2(sint1, cost1)

    program = Program()
    program.inst(gt.RZ(keep_within_limits(t0+b[2]*py.pi), qubit_index))
    program.inst(gt.RX(+s[0]*py.pi/2, qubit_index))
    program.inst(gt.RZ(keep_within_limits(s[0]*s[2]*t1), qubit_index))
    program.inst(gt.RX(-s[1]*py.pi/2, qubit_index))
    program.inst(gt.RZ(keep_within_limits(t2+b[2]*py.pi), qubit_index))

    return program
#
# ##############################################
#
#     for a_xor_b in [True,False]:
#         if a_xor_b:
#             t0Pt2by2 = math.atan2(u3.real,u0.real)
#             t0Mt2by2 = math.atan2(u1.real,u2.real)
#             cost1 = py.sqrt(u0.real**2 + u3.real**2) / 2.
#             sint1 = py.sqrt(u1.real**2 + u2.real**2) / 2.
#         else:
#             t0Pt2by2 = math.atan2(-u0.real,u3.real)
#             t0Mt2by2 = math.atan2(-u2.real,u1.real)
#             sint1 = py.sqrt(u0.real**2 + u3.real**2) / 2.
#             cost1 = py.sqrt(u1.real**2 + u2.real**2) / 2.
#         t0 = (t0Pt2by2 + t0Mt2by2)
#         t2 = (t0Pt2by2 - t0Mt2by2)
#         t1 = 2 * math.atan2(sint1, cost1)
#         for x_correction in [False,True]:
#             for z_correction in [False,True]:
#                 angles = [t0, +py.pi/2, t1, -py.pi/2, t2]
#                 if not a_xor_b:
#                     angles[3] *= -1
#                 if z_correction:
#                     angles[0] += - py.pi
#                     angles[4] += + py.pi
#                     angles[2] *= -1
#                 if x_correction:
#                     angles[1] *= -1
#                     angles[3] *= -1
#                     angles[2] *= -1
#                 angles = [keep_within_limits(x) for x in angles]
#                 program = Program()
#                 program.inst(gt.RZ(angles[0], index))
#                 program.inst(gt.RX(angles[1], index))
#                 program.inst(gt.RZ(angles[2], index))
#                 program.inst(gt.RX(angles[3], index))
#                 program.inst(gt.RZ(angles[4], index))
#                 yield program

def convert_unitary_to_bloch(unitary):
    u0 = ((unitary).tr() / 2.0)
    u1 = ((unitary*qp.sigmax()).tr() * 1j / 2.0)
    u2 = ((unitary*qp.sigmay()).tr() * 1j / 2.0)
    u3 = ((unitary*qp.sigmaz()).tr() * 1j / 2.0)
    return (u0,u1,u2,u3)

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

class CliffordCorrection(TwoQubitCompilation):

    clifford_operators = {
        0: ('ZXY', qp.qeye(2)),
        1: ('ZYX', (qp.sigmax() + qp.sigmay()) / py.sqrt(2)),
        2: ('XZY', (qp.sigmaz() + qp.sigmax()) / py.sqrt(2)),
        3: ('XYZ', (-1j*qp.qeye(2) - qp.sigmaz() - qp.sigmax() + qp.sigmay())/2),
        4: ('YXZ', (qp.sigmay() + qp.sigmaz()) / py.sqrt(2)),
        5: ('YZX', (-1j*qp.qeye(2) - qp.sigmaz() - qp.sigmax() - qp.sigmay())/2),
    }

    def apply_Clifford_correction(self, index):
        correction, operator  = self.clifford_operators[index]
        params_original = {x:y for x,y in zip('ZXY', self.standard_params)}
        self.clifford_index = index
        self.clifford_params = tuple([params_original[x] for x in correction])
        self.clifford_W0 = operator * self.standard_W0
        self.clifford_W1 = operator * self.standard_W1
        self.clifford_W2 = self.standard_W2 * operator.dag()
        self.clifford_W3 = self.standard_W3 * operator.dag()

class UsingCPhase(CliffordCorrection):

    def gen_twoqubitcircuit(self, index):
        self.twoqubit_index = index
        b = [(self.twoqubit_index & (1<<(n)))>>(n) for n in range(10)[::-1]]
        s = [+1 if x==0 else -1 for x in b]

        alpha_corr = s[5] * self.clifford_params[0]
        beta_corr  = s[6] * self.clifford_params[1]
        gamma_corr = s[4] * self.clifford_params[2]

        W0 = rotation_z(-gamma_corr+(b[0]+b[7]+b[8])*py.pi) \
            * rotation_x(-s[9]*s[8]*py.pi/2)
        W1 = rotation_z(-gamma_corr+(b[1]+b[4]+b[5]+b[7]+b[8])*py.pi) \
            * rotation_x(+s[9]*s[8]*s[4]*py.pi/2)
        W2 = rotation_y(+s[8]*py.pi/2) \
            * rotation_z(-beta_corr+py.pi/2+(b[2]+b[7]+b[8]+b[9])*py.pi)
        W3 = rotation_y(-s[8]*s[6]*py.pi/2) \
            * rotation_z(-beta_corr+py.pi/2+(b[3]+b[5]+b[6]+b[7]+b[8]+b[9])*py.pi)
        self.pauli_W0 = W0 * self.clifford_W0
        self.pauli_W1 = W1 * self.clifford_W1
        self.pauli_W2 = self.clifford_W2 * W2
        self.pauli_W3 = self.clifford_W3 * W3

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

        self.pauli_program = program

    def gen_twoqubitcircuit_OLD_CORRECT(self, index):
        self.twoqubit_index = index

        ky = self.convert_to_pm1(self.select_base4(index, 4))
        kx = self.convert_to_pm1(self.select_base4(index, 3))
        kz = self.convert_to_pm1(self.select_base4(index, 2))
        p0 = self.convert_to_pm1(self.select_base4(index, 1))
        p1 = self.convert_to_pm1(self.select_base4(index, 0))

        gamma_corr = ky[0] * ky[1] * p0[0] * p1[0] * p0[1] * p1[1] * self.clifford_params[2]
        alpha_corr = p0[0] * p1[0] * self.clifford_params[0]
        beta_corr = p0[1] * p1[1] * kx[0] * kx[1] * kz[0] * kz[1] * self.clifford_params[1]

        angle_x1 = lambda s: beta_corr + py.pi/2 if s>0 else beta_corr - py.pi/2
        angle_z1 = lambda s: alpha_corr - py.pi/2 if s>0 else alpha_corr + py.pi/2

        # W0 = rotation_z(gamma_corr) * rotation_x(-ky[0]*py.pi/2) * self.paulis[p0]
        # W1 = rotation_z(gamma_corr) * rotation_x(-ky[1]*py.pi/2) * self.paulis[p1]
        W0 = rotation_z(gamma_corr+py.pi*(1-p0[1])/2) * rotation_x(-ky[0]*p0[0]*p0[1]*py.pi/2)
        W1 = rotation_z(gamma_corr+py.pi*(1-p1[1])/2) * rotation_x(-ky[1]*p1[0]*p1[1]*py.pi/2)
        # W2 = self.paulis[p0] * rotation_y(+kx[0]*kz[0]*py.pi/2) * rotation_z(angle_x1(kz[0]))
        # W3 = self.paulis[p1] * rotation_y(+kx[1]*kz[1]*py.pi/2) * rotation_z(angle_x1(kz[1]))
        W2 = rotation_y(+kx[0]*kz[0]*p0[1]*py.pi/2) * rotation_z(angle_x1(kz[0])+py.pi*(1-p0[0]*p0[1])/2)
        W3 = rotation_y(+kx[1]*kz[1]*p1[1]*py.pi/2) * rotation_z(angle_x1(kz[1])+py.pi*(1-p1[0]*p1[1])/2)

        program = Program()
        program.inst(gt.CPHASE(keep_within_limits(-2*gamma_corr), self.qubit_index[0], self.qubit_index[1]))
        for i in range(2):
            program.inst(gt.RX(+ky[i]*py.pi/2, self.qubit_index[i]))
        for i in range(2):
            program.inst(gt.RZ(keep_within_limits(angle_z1(kz[i])), self.qubit_index[i]))
        program.inst(gt.CPHASE(keep_within_limits(-2*alpha_corr), self.qubit_index[0], self.qubit_index[1]))
        for i in range(2):
            program.inst(gt.RX(-kx[i]*py.pi/2, self.qubit_index[i]))
        program.inst(gt.CPHASE(keep_within_limits(-2*beta_corr), self.qubit_index[0], self.qubit_index[1]))

        self.pauli_W0 = W0 * self.clifford_W0
        self.pauli_W1 = W1 * self.clifford_W1
        self.pauli_W2 = self.clifford_W2 * W2
        self.pauli_W3 = self.clifford_W3 * W3
        self.pauli_program = program

# paulis = {
#     (+1,+1): qp.qeye(2),
#     (+1,-1): qp.sigmaz(),
#     (-1,+1): qp.sigmax(),
#     (-1,-1): qp.sigmay()
# }
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
