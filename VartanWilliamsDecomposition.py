"""
Author: Tiago Batalhão

This file contains code to implement a 2-qubit unitary operator
according to the method of # XXX:
"""

import pylab as py
import qutip as qp
import cmath
import scipy.optimize
from EquivalenceClasses import equivalence_partition

class TwoQubitOperation():
    """
    This top-class will implement the methods described in PRA 63, 062309.
    It uses the qutip package.
    """

    def decomposition_Cirac(self):
        """
        Decompose the unitary according to the method in PRA 63, 062309.
        """
        # Step (i)
        utu = calculate_UtU(self.unitary)
        egvals, entangled_A = find_entangled_basis(utu)
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
        two_qubit_params = list(get_alphabetagamma(Udiag))

        self._Cirac_unitaries = tuple([Vb,Va,Ub,Ua])
        self._Cirac_parameters = two_qubit_params

    def reconstruct(self):
        unitaries = self._Cirac_unitaries
        params = self._Cirac_parameters
        basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
        W4 = (-0.5j*sum([x*y for x,y in zip(params,basis)])).expm()
        _reconstruct = qp.tensor([unitaries[1],unitaries[0]])
        _reconstruct = W4 * _reconstruct
        _reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * _reconstruct
        return _reconstruct

def calculate_UtU(operator):
    """
    Calcutate U^T U, where U is the transpose of U
    in the magical basis.
    """
    operator = qp.Qobj(operator)
    operator_magical = convert_to_magicalbasis(operator)
    transpose = operator_magical.full().T
    transpose = qp.Qobj(transpose,dims=operator.dims)
    utu = (transpose * operator_magical)
    return convert_from_magicalbasis(utu)

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

def finding_local_unitaries(entangled_basis):
    """
    Find Va, Vb such that
        (Va \otimes Vb) \ket{\Psi_k} = \ket{\Phi_k}
    where
        \ket{\Psi_k} is an element of the input basis
        \ket{\Phi_k} is an element of the magical basis

    Input:
        entangled_basis: a basis with four maximally-entangled states
    """

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
    changed_order = [_magical_vectors[x] for x in [0,1,3,2]]
    for vector,magical in zip(entangled_basis,changed_order):
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

def make_vector_real(vector):
    """
    Extract a common complex factor from a vector.
    """
    threshold_A = 1e-6
    phases = [cmath.phase(x)%py.pi for x in vector if abs(x)>threshold_A]
    diff = py.diff(phases + phases[0:1])
    threshold_B = 1e-2
    assert min(abs(diff)) < threshold_B , u"Vector can not be made real"
    phase = phases[0]
    vector = vector * py.exp(-1j*phase)
    return vector, phase


def find_entangled_basis(operator):
    """
    Find an entangled basis for a given operator.
    Must deal with degenerate subspaces.
    """
    eigenvals, eigenvecs = operator.eigenstates()

    # Find the degenerate eigenspaces
    threshold = 1e-10
    degenerate = equivalence_partition(eigenvals, lambda x,y: abs(x-y)<threshold)
    subspaces = [(x,[]) for x in degenerate[0]]
    for key, value in degenerate[1].items():
        subspaces[value][1].append(eigenvecs[key])

    eigenvalues, eigenvectors = [], []
    for subspace in subspaces:
        basis = find_entangled_elements(sum(qp.ket2dm(x) for x in subspace[1]))
        for ket in basis:
            eigenvalues.append(subspace[0])
            eigenvectors.append(ket)

    # QUICK TEST (ERASE):
    reconstruct = sum(x*qp.ket2dm(y) for x,y in zip(eigenvalues,eigenvectors))
    from test_VartanWilliamsDecomposition import unitary_equivalence
    test = unitary_equivalence(reconstruct, operator)
    assert test, u"Failure at find_entangled_basis"

    print(eigenvectors)

    return eigenvalues, eigenvectors


def find_entangled_elements(projector):
    """
    From a list of vectors, find a basis to the spanned space
    formed by maximally-entangled states.
    """
    basis = [y for x,y in zip(*projector.eigenstates()) if abs(x)>1e-4]
    print(len(basis))
    if len(basis) < 2:
        return basis
    else:
        def attempt(params):
            params_real = params[:len(basis)]
            params_imag = params[len(basis):]
            ket = sum((x+1j*y)*z for x,y,z in zip(params_real, params_imag, basis))
            rho = qp.ket2dm(ket)
            normalization = rho.tr()
            ket = ket / py.sqrt(normalization)
            rho = qp.ket2dm(ket)
            return ket, rho
        def function_to_minimize(params):
            ket, rho = attempt(params)
            rho_A = qp.ptrace(rho,[0])
            metric = - min(rho_A.eigenenergies())
            return metric
        initial_attempt = [0]*(2*len(basis))
        optimal = scipy.optimize.fmin(function_to_minimize, initial_attempt)
        optimal_ket, optimal_rho = attempt(optimal)
        assert (projector * optimal_rho * projector - optimal_rho).norm() < 1e-4, u"rho is not in the subspace of projector"
        projector_next = projector - optimal_rho
        print(optimal_rho.eigenenergies())
        print(projector.eigenenergies())
        print(projector_next.eigenenergies())
        return [optimal_ket] + find_entangled_elements(projector_next)
