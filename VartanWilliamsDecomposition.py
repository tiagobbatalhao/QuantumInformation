"""
Author: Tiago Batalhão

This file contains code to implement a 2-qubit unitary operator
according to the method of PRA 63, 062309.
"""

import pylab as py
import qutip as qp
import cmath
import scipy.optimize
from EquivalenceClasses import equivalence_partition
import random

class TwoQubitOperation():
    """
    This top-class will implement the methods described in PRA 63, 062309.
    It uses the qutip package.
    """

    def __init__(self, unitary=None):
        if unitary is not None:
            self.unitary = unitary

    @property
    def unitary(self):
        return self.__unitary.full()

    @unitary.setter
    def unitary(self, unitary):
        if isinstance(unitary, list) or isinstance(unitary, tuple):
            unitary = py.array(unitary)
        if hasattr(unitary, 'shape') and unitary.shape == (4,4):
            self.__unitary = qp.Qobj(unitary, dims=[[2,2],[2,2]])
            self.decomposition()
        else:
            raise Exception("Property 'unitary' must be 4x4 matrix.")

    @property
    def Cirac_parameters(self):
        parameters = []
        for unitary in self.__Cirac_parameters[0:4]:
            parameters.append(format_unitary_by_trace(unitary))
        parameters.append(self.__Cirac_parameters[-1])
        return tuple(parameters)

    @Cirac_parameters.setter
    def Cirac_parameters(self, value):
        raise Exception("You can not assign to this property.")

    def decomposition(self):
        """
        Decompose the unitary according to the method in PRA 63, 062309.
        """
        # Step (i)
        unitary = qp.Qobj(self.unitary, dims=[[2,2],[2,2]])
        egvals, entangled_A, magical_A = find_entangled_basis_magical(unitary)
        epsilons = [0.5*(cmath.phase(x)%(2*py.pi)) for x in egvals]
        # Step (ii)
        Va, Vb, xis = finding_local_unitaries(magical_A)
        # Step (iii)
        entangled_B = [py.exp(-1j*epsilon)*unitary*ket for ket,epsilon in zip(entangled_A,epsilons)]
        magical_B = [convert_to_magicalbasis(x) for x in entangled_B]
        # Step (iv)
        Uad, Ubd, phases = finding_local_unitaries(magical_B)
        Ua = Uad.dag()
        Ub = Ubd.dag()
        Udiag = qp.tensor([Uad,Ubd]) * unitary * qp.tensor([Va,Vb]).dag()
        two_qubit_params = list(get_alphabetagamma(Udiag))

        two_qubit_params = tuple(x if abs(x)>1e-12 else 0.0 for x in two_qubit_params)
        self.__Cirac_parameters = tuple([Vb, Va, Ub, Ua, two_qubit_params])

    def reconstruct(self):
        """
        Return the expression of the unitary as a product of the decomposition.
        It is the inverse of the decomposition. Useful for testing.
        """
        unitaries = [qp.Qobj(x,dims=[[2],[2]]) for x in self.Cirac_parameters[0:4]]
        params = self.Cirac_parameters[-1]
        basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
        W4 = (-0.5j*sum([x*y for x,y in zip(params,basis)])).expm()
        _reconstruct = qp.tensor([unitaries[1],unitaries[0]])
        _reconstruct = W4 * _reconstruct
        _reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * _reconstruct
        return _reconstruct.full()

def find_entangled_basis_magical(operator):
    """
    Find a basis formed from maximally-entangled elements
    of the operator U^T U, where U is the transpose of U
    in the magical basis.
    """
    operator = qp.Qobj(operator, dims=[[2,2],[2,2]])
    operator_magical = convert_to_magicalbasis(operator)
    transpose = operator_magical.full().T
    transpose = qp.Qobj(transpose,dims=operator.dims)
    utu = (transpose * operator_magical)
    egvals, magical = utu.eigenstates()
    entangled = [convert_from_magicalbasis(x) for x in magical]
    return egvals, entangled, magical

def convert_to_magicalbasis(operator):
    """
    Convert an operator from the computational basis
    to the magical basis.
    """
    if operator.type == 'oper':
        return _magical_basis_dag * operator * _magical_basis
    if operator.type == 'ket':
        return _magical_basis_dag * operator
    if operator.type == 'bra':
        return operator * _magical_basis

def convert_from_magicalbasis(operator):
    """
    Convert an operator from the magical basis
    to the computational basis.
    """
    if operator.type == 'oper':
        return _magical_basis * operator * _magical_basis_dag
    if operator.type == 'ket':
        return _magical_basis * operator
    if operator.type == 'bra':
        return operator * _magical_basis_dag

_magical_vectors = [
    [  1,  0,  0,  1],
    [-1j,  0,  0, 1j],
    [  0,  1, -1,  0],
    [  0,-1j,-1j,  0],
]
_magical_basis = qp.Qobj(py.array(_magical_vectors).T/py.sqrt(2),dims=[[2,2],[2,2]])
_magical_basis_dag = _magical_basis.dag()

def finding_local_unitaries(in_magical_basis):
    """
    Find Va, Vb such that
        (Va \otimes Vb) \ket{\Psi_k} = \ket{\Phi_k}
    where
        \ket{\Psi_k} is an element of the input basis
        \ket{\Phi_k} is an element of the magical basis

    Input:
        in_magical_basis: a basis with real vectors, representing
            maximally-entangled states in the magical basis.
    """

    ket_ef = convert_from_magicalbasis(in_magical_basis[0] + 1j * in_magical_basis[1]) / py.sqrt(2)
    ket_epfp = convert_from_magicalbasis(in_magical_basis[0] - 1j * in_magical_basis[1]) / py.sqrt(2)

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
    for base in in_magical_basis:
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
    changed_order = [_magical_vectors[x] for x in [0,1,3,2]]
    entangled_basis = [convert_from_magicalbasis(x) for x in in_magical_basis]
    for vector, magical in zip(entangled_basis, changed_order):
        magical_ket = qp.Qobj(py.array(magical).T,dims=[[2,2],[1,1]]) / py.sqrt(2)
        vec = transform * vector
        inner_product = identity.matrix_element(magical_ket.dag(),vec)
        phase = - cmath.phase(inner_product)
        phases.append( phase%(2*py.pi) )

    return unitary_A, unitary_B, phases

def get_alphabetagamma(operator):
    """
    Write an operator in the form
        U = exp(-0.5j * (alpha*ZZ + beta * XX + gamma * YY))
    It's not checked if the operator is in this form
    """
    eigenvalues, eigenstates = operator.eigenstates()
    phases = [cmath.phase(x) for x in eigenvalues]
    hamiltonian = -sum([x*qp.ket2dm(y) for x,y in zip(phases,eigenstates)])
    hamiltonian = qp.Qobj(hamiltonian,dims = [[2,2],[2,2]])
    alpha = qp.expect(hamiltonian, qp.tensor([qp.sigmaz()]*2)) / 2.0
    beta = qp.expect(hamiltonian, qp.tensor([qp.sigmax()]*2)) / 2.0
    gamma = qp.expect(hamiltonian, qp.tensor([qp.sigmay()]*2)) / 2.0
    return alpha, beta, gamma

def format_unitary_by_det(unitary):
    """
    Convert a single-qubit unitary to numpy array
    with determinant +1 and imaginary trace.
    """
    matrix = unitary.tidyup()
    determinant = py.det(matrix.full())
    matrix = matrix / py.sqrt(determinant)
    trace = py.trace(matrix.full())
    if trace.real > +1e-12:
        matrix = +1* matrix
    elif trace.real < -1e-12:
        matrix = -1 * matrix
    elif trace.imag > +1e-12:
        matrix = -1j * matrix
    elif trace.imag < -1e-12:
        matrix = +1j * matrix
    return matrix.tidyup().full()

def format_unitary_by_trace(unitary):
    """
    Convert a single-qubit unitary to numpy array
    with determinant +1 and imaginary trace.
    """
    trace = py.trace(unitary.full())
    if abs(trace) > 1e-12:
        correction = py.exp(-1j* cmath.phase(trace))
        matrix = correction * unitary
    elif abs(unitary[0,0]) > 1e-12:
        correction = py.exp(-1j* cmath.phase(unitary[0,0]))
        matrix = -1j * correction * unitary
    else:
        phases = [
            cmath.phase(unitary[0,1]),
            cmath.phase(unitary[1,0]),
        ]
        mid_phase = sum(phases) / 2.0
        correction = py.exp(-1j * mid_phase)
        matrix = -1j * correction * unitary
    return matrix.tidyup().full()

class TwoQubitOperation_Clifford(TwoQubitOperation):
    """
    This class will implement the different ways to perform a 2-qubit operation
    according to Clifford corrections.
    It uses the qutip package.
    """

    @property
    def Clifford_parameters(self):
        implementations = []
        for implementation in self.__Clifford_parameters:
            parameters = []
            for unitary in implementation[0:4]:
                parameters.append(format_unitary_by_trace(unitary))
            parameters.append(implementation[-1])
            implementations.append(parameters)
        return implementations

    @Clifford_parameters.setter
    def Clifford_parameters(self, value):
        raise Exception("You can not assign to this property.")

    def decomposition(self):
        """
        Create 96 distinct ways of performing a 2-qubit operation.
        """
        super().decomposition()

        old_tuple = correct_alphabetagamma(self.Cirac_parameters)

        implementations = []
        clifford_order = ['ZXY', 'ZYX', 'XZY', 'XYZ', 'YZX', 'YXZ']
        pauli_order = ['I', 'Z', 'X', 'Y']
        for clifford_label in clifford_order:
            clifford_ops = correction_clifford[clifford_label]
            for paulis_label in pauli_order:
                paulis_ops = correction_pauli_single[paulis_label]
                for paulid_label in pauli_order:
                    paulid_ops = correction_pauli_double[paulid_label]
                    new_tuple = apply_corrections(old_tuple, clifford_ops)
                    new_tuple = apply_corrections(new_tuple, paulis_ops)
                    new_tuple = apply_corrections(new_tuple, paulid_ops)
                    implementations.append(new_tuple)
        self.__Clifford_parameters = tuple(implementations)

    def reconstruct(self, index=0):
        """
        Return the expression of the unitary as a product of the decomposition.
        Indexed by the Clifford correction
        """
        unitaries = [qp.Qobj(x,dims=[[2],[2]]) for x in self.Clifford_parameters[index][0:4]]
        params = self.Clifford_parameters[index][-1]
        basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
        W4 = (-0.5j*sum([x*y for x,y in zip(params,basis)])).expm()
        _reconstruct = qp.tensor([unitaries[1],unitaries[0]])
        _reconstruct = W4 * _reconstruct
        _reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * _reconstruct
        return _reconstruct.full()


def correct_alphabetagamma(Cirac_parameters):
    """
    Change the implementation so that the parameters alpha,beta,gamma
    are in decreasing order of absolute value and
    alpha and beta are positives.
    """
    new_tuple = [qp.Qobj(x,dims=[[2],[2]]) for x in Cirac_parameters[:4]]
    new_tuple.append(list(Cirac_parameters[-1]))

    # Put each element in the range (-pi/2, +pi/2)
    indices = range(3)
    sigmas = [qp.sigmaz(), qp.sigmax(), qp.sigmay()]
    for index, sigma in zip(indices, sigmas):
        correction = False
        while new_tuple[-1][index] > +py.pi/2:
            new_tuple[-1][index] -= py.pi
            correction = not correction
        while new_tuple[-1][index] <= -py.pi/2:
            new_tuple[-1][index] += py.pi
            correction = not correction
        if correction:
            new_tuple[0] = sigma * new_tuple[0]
            new_tuple[1] = sigma * new_tuple[1]


    order = sorted(enumerate(new_tuple[-1]), key=lambda x:-abs(x[1]))
    pauli_basis = 'ZXY'
    correction = ''.join(pauli_basis[x[0]] for x in order)
    new_tuple = apply_corrections(new_tuple, correction_clifford[correction])
    if new_tuple[-1][0] < 0:
        new_tuple = apply_corrections(new_tuple, correction_pauli_single['X'])
    if new_tuple[-1][1] < 0:
        new_tuple = apply_corrections(new_tuple, correction_pauli_single['Z'])

    return new_tuple

def apply_corrections(original, corrections):
    """
    Apply a correction to the Cirac parameters
    Input:
        original: a 5-tuple. The first 4 are unitaries, the last is a tuple.
        parameters: a 3-tuple. The first 2 are unitaries, the last is a function.

    The correction is
        V_0 = C_0 * U_0
        V_1 = C_1 * U_1
        V_2 = U_2 * C_0^d
        V_3 = U_3 * C_1^d
    """
    new = []
    new.append(corrections[0] * original[0])
    new.append(corrections[1] * original[1])
    new.append(original[2] * corrections[0].dag())
    new.append(original[3] * corrections[1].dag())
    new.append(tuple(corrections[2](original[4])))
    return tuple(new)



def define_corrections():
    """
    Define the Clifford corrections.
    """
    clifford = {}
    clifford['ZXY'] = (
        qp.qeye(2),
        qp.qeye(2),
        lambda x: tuple([+x[0], +x[1], +x[2]])
    )
    clifford['XZY'] = (
        (qp.sigmaz() + qp.sigmax()) / py.sqrt(2),
        (qp.sigmaz() + qp.sigmax()) / py.sqrt(2),
        lambda x: tuple([+x[1], +x[0], +x[2]])
    )
    clifford['YXZ'] = (
        (qp.sigmaz() + qp.sigmay()) / py.sqrt(2),
        (qp.sigmaz() + qp.sigmay()) / py.sqrt(2),
        lambda x: tuple([+x[2], +x[1], +x[0]])
    )
    clifford['ZYX'] = (
        (qp.sigmax() + qp.sigmay()) / py.sqrt(2),
        (qp.sigmax() + qp.sigmay()) / py.sqrt(2),
        lambda x: tuple([+x[0], +x[2], +x[1]])
    )
    clifford['YZX'] = (
        (+1j* qp.qeye(2) + qp.sigmaz() + qp.sigmax() + qp.sigmay()) / 2.,
        (+1j* qp.qeye(2) + qp.sigmaz() + qp.sigmax() + qp.sigmay()) / 2.,
        lambda x: tuple([+x[2], +x[0], +x[1]])
    )
    clifford['XYZ'] = (
        (-1j* qp.qeye(2) + qp.sigmaz() + qp.sigmax() + qp.sigmay()) / 2.,
        (-1j* qp.qeye(2) + qp.sigmaz() + qp.sigmax() + qp.sigmay()) / 2.,
        lambda x: tuple([+x[1], +x[2], +x[0]])
    )

    pauli_double = {}
    pauli_double['I'] = (
        qp.qeye(2),
        qp.qeye(2),
        lambda x: tuple([+x[0], +x[1], +x[2]])
    )
    pauli_double['Z'] = (
        qp.sigmaz(),
        qp.sigmaz(),
        lambda x: tuple([+x[0], +x[1], +x[2]])
    )
    pauli_double['X'] = (
        qp.sigmax(),
        qp.sigmax(),
        lambda x: tuple([+x[0], +x[1], +x[2]])
    )
    pauli_double['Y'] = (
        qp.sigmay(),
        qp.sigmay(),
        lambda x: tuple([+x[0], +x[1], +x[2]])
    )

    pauli_single = {}
    pauli_single['I'] = (
        qp.qeye(2),
        qp.qeye(2),
        lambda x: tuple([+x[0], +x[1], +x[2]])
    )
    pauli_single['Z'] = (
        qp.qeye(2),
        qp.sigmaz(),
        lambda x: tuple([+x[0], -x[1], -x[2]])
    )
    pauli_single['X'] = (
        qp.qeye(2),
        qp.sigmax(),
        lambda x: tuple([-x[0], +x[1], -x[2]])
    )
    pauli_single['Y'] = (
        qp.qeye(2),
        qp.sigmay(),
        lambda x: tuple([-x[0], -x[1], +x[2]])
    )

    return clifford, pauli_single, pauli_double

corrections = ['correction_clifford', 'correction_pauli_single', 'correction_pauli_double']
if any(x not in locals() for x in corrections):
    correction_clifford, correction_pauli_single, correction_pauli_double = define_corrections()
