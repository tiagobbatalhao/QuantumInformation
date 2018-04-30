import numpy as py
import unittest
import cmath
import random
import UsingCPhase as mod
from pyquil.quil import Program
import pyquil.gates as gt
import qutip as qp

class QVM_get_unitary_matrix(unittest.TestCase):

    def test_hadamard(self):
        correct = py.array([[1,1],[1,-1]]) / py.sqrt(2)
        program = Program()
        program.inst(gt.H(0))
        attempt = mod.QVM_get_unitary_matrix(program,[0])
        difference = attempt - correct
        norm = sum(abs(difference.flatten()))
        self.assertLess(norm, 1e-6)

    def test_CNOT_01(self):
        correct = py.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
        program = Program()
        program.inst(gt.H(1))
        program.inst(gt.CZ(0,1))
        program.inst(gt.H(1))
        attempt = mod.QVM_get_unitary_matrix(program,[0,1])
        difference = attempt - correct
        norm = sum(abs(difference.flatten()))
        self.assertLess(norm, 1e-6)

    def test_CNOT_10(self):
        correct = py.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        program = Program()
        program.inst(gt.H(0))
        program.inst(gt.CZ(0,1))
        program.inst(gt.H(0))
        attempt = mod.QVM_get_unitary_matrix(program,[0,1])
        difference = attempt - correct
        norm = sum(abs(difference.flatten()))
        self.assertLess(norm, 1e-6)

class two_qubit_unitary(unittest.TestCase):

    def setUp(self):
        self.basis = []
        self.basis.append(qp.tensor([qp.sigmaz()]*2))
        self.basis.append(qp.tensor([qp.sigmax()]*2))
        self.basis.append(qp.tensor([qp.sigmay()]*2))

    def unitary_equivalence(self, uA, uB):
        thisCorrect = True
        should_be_identity = uA.dag() * uB
        eigs = should_be_identity.eigenenergies()
        eigs*= py.exp(-1j*cmath.phase(eigs[0]))
        for eig in eigs:
            if abs(eig-1) > 1e-6:
                thisCorrect = False
                break
        return thisCorrect

    @unittest.skip('Too slow. Already tested.')
    def test_two_qubit_unitary(self):
        params = [2*py.pi*random.random() for x in range(3)]
        unitary = sum([-0.5j*x*y for x,y in zip(params,self.basis)]).expm()
        successes = 0
        for counter,impl in enumerate(mod.two_qubit_unitary(*params)):
            w0, w1, w2, w3, program = impl
            w4 = mod.QVM_get_unitary_matrix(program,[0,1])
            reconstruct = qp.tensor([w1,w0])
            reconstruct = qp.Qobj(w4,dims=[[2,2],[2,2]]) * reconstruct
            reconstruct = qp.tensor([w3,w2]) * reconstruct
            thisCorrect = self.unitary_equivalence(reconstruct,unitary)
            if thisCorrect:
                successes += 1
            print('{:d} successes in {:d} tries'.format(successes,counter+1))
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)

    # @unittest.skip('Too slow. Already tested.')
    def test_apply_pauli_corrections_V2(self):
        params = [2*py.pi*random.random() for x in range(3)]
        unitary = sum([-0.5j*x*y for x,y in zip(params,self.basis)]).expm()
        successes = 0
        attempts = mod.apply_pauli_corrections_V2(*params)
        random.shuffle(attempts)
        for counter,impl in enumerate(attempts):
            w0, w1, w2, w3, program = impl
            w4 = mod.QVM_get_unitary_matrix(program,[0,1])
            reconstruct = qp.tensor([w1,w0])
            reconstruct = qp.Qobj(w4,dims=[[2,2],[2,2]]) * reconstruct
            reconstruct = qp.tensor([w3,w2]) * reconstruct
            thisCorrect = self.unitary_equivalence(reconstruct,unitary)
            if thisCorrect:
                successes += 1
            print('{:d} successes in {:d} tries'.format(successes,counter+1))
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)

    @unittest.skip('Too slow. Already tested.')
    def test_apply_clifford_corrections(self):
        params = [2*py.pi*random.random() for x in range(3)]
        unitary = sum([-0.5j*x*y for x,y in zip(params,self.basis)]).expm()
        successes = 0
        for counter,impl in enumerate(mod.apply_clifford_corrections(*params)):
            w0, w1, w2, w3, program = impl
            w4 = mod.QVM_get_unitary_matrix(program,[0,1])
            reconstruct = qp.tensor([w1,w0])
            reconstruct = qp.Qobj(w4,dims=[[2,2],[2,2]]) * reconstruct
            reconstruct = qp.tensor([w3,w2]) * reconstruct
            thisCorrect = self.unitary_equivalence(reconstruct,unitary)
            if thisCorrect:
                successes += 1
            print('{:d} successes in {:d} tries'.format(successes,counter+1))
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)
