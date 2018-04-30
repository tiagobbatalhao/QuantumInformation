import numpy as py
import unittest
import cmath
import random
import TwoQubitCompilation as mod
from pyquil.quil import Program
import pyquil.api as api
import pyquil.gates as gt
qvm = api.QVMConnection()
import qutip as qp

class test_TwoQubitCompilation(unittest.TestCase):

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
        compiled = mod.UsingCPhase(unitary)
        j_shuffled = list(range(2**22))
        random.shuffle(j_shuffled)
        successes, failures = 0,0
        for j in j_shuffled:
            print(j)
            circuit = compiled.compile(j,3)[(j,3)]
            reconstruct = QVM_get_unitary_matrix(circuit,compiled.qubit_index)
            thisCorrect = self.unitary_equivalence(qp.Qobj(reconstruct),unitary)
            if thisCorrect:
                successes += 1
            else:
                failures += 1
            print('So far, {} successes and {} failures. Tested j = {}'.format(successes,failures,j))
            with self.subTest(instance = j):
                self.assertTrue(thisCorrect)

    @unittest.skip('Already tested')
    def test_decomposition_1Q_gates(self):
        unitary = qp.rand_unitary(2)
        # circuits = list(mod.decomposition_1Q_gates(unitary))
        # for i,circuit in enumerate(circuits):
        for i in range(8):
            circuit = mod.decomposition_1Q_gates(unitary,i)
            reconstruct = QVM_get_unitary_matrix(circuit,[0])
            thisCorrect = self.unitary_equivalence(qp.Qobj(reconstruct),unitary)
            with self.subTest(instance = i):
                self.assertTrue(thisCorrect)

    def test_decompositionCirac(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.TwoQubitCompilation(unitary)
        cirac = compiled.decomposition_Cirac()
        reconstruct = qp.tensor(*cirac[2:4]) * cirac[4] * qp.tensor(*cirac[0:2])
        # reconstruct = qp.tensor(*cirac[3:1:-1]) * cirac[4] * qp.tensor(*cirac[1::-1])
        thisCorrect = self.unitary_equivalence(reconstruct,compiled.unitary)
        self.assertTrue(thisCorrect)
        reconstruct_2 = qp.tensor([compiled.standard_W1,compiled.standard_W0])
        reconstruct_2 = compiled.standard_W4 * reconstruct_2
        reconstruct_2 = qp.tensor([compiled.standard_W3,compiled.standard_W2]) * reconstruct_2
        thisCorrect = self.unitary_equivalence(reconstruct_2,compiled.unitary)
        self.assertTrue(thisCorrect)

    def test_alphabetagamma(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.TwoQubitCompilation(unitary)
        cirac = compiled.decomposition_Cirac()
        params = compiled.get_alphabetagamma()
        hamiltonian = sum([x*qp.tensor([eval('qp.sigma'+y+'()')]*2) for x,y in zip(params,'zxy')])
        reconstruct = (-0.5j*hamiltonian).expm()
        thisCorrect = self.unitary_equivalence(reconstruct,compiled.standard_W4)
        self.assertTrue(thisCorrect)

    def test_CliffordCorrection(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.CliffordCorrection(unitary)
        for i in range(6):
            compiled.apply_Clifford_correction(i)
            params = compiled.clifford_params
            hamiltonian = sum([x*qp.tensor([eval('qp.sigma'+y+'()')]*2) for x,y in zip(params,'zxy')])
            reconstruct_W4 = (-0.5j*hamiltonian).expm()
            reconstruct = qp.tensor([compiled.clifford_W1,compiled.clifford_W0])
            reconstruct = reconstruct_W4 * reconstruct
            reconstruct = qp.tensor([compiled.clifford_W3,compiled.clifford_W2]) * reconstruct
            thisCorrect = self.unitary_equivalence(reconstruct,compiled.unitary)
            with self.subTest(instance = i):
                self.assertTrue(thisCorrect)

    @unittest.skip('Not yet there')
    def test_PauliCorrection(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.UsingCPhase(unitary)
        successes = 0
        failures = 0
        j_shuffled = list(range(1024))
        random.shuffle(j_shuffled)
        for i in range(6):
            compiled.apply_Clifford_correction(i)
            for j in j_shuffled:
                compiled.gen_twoqubitcircuit(j)

                reconstruct_W4 = QVM_get_unitary_matrix(compiled.pauli_program)
                reconstruct_W4 = qp.Qobj(reconstruct_W4, dims=[[2,2],[2,2]])
                reconstruct = qp.tensor([compiled.pauli_W1,compiled.pauli_W0])
                reconstruct = reconstruct_W4 * reconstruct
                reconstruct = qp.tensor([compiled.pauli_W3,compiled.pauli_W2]) * reconstruct
                thisCorrect = self.unitary_equivalence(reconstruct,compiled.unitary)
                if thisCorrect:
                    successes += 1
                else:
                    failures += 1
                print('So far, {} successes and {} failures. Tested j = {}'.format(successes,failures,j))

                with self.subTest(clifford = i, pauli = j):
                    self.assertTrue(thisCorrect)



def QVM_get_unitary_matrix(program, indices = [0,1]):
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
