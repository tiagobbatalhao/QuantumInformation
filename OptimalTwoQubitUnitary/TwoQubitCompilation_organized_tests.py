import numpy as py
import qutip as qp
import cmath
import unittest, random
from pyquil.quil import Program
import pyquil.gates as gt
import pyquil.api as api
qvm = api.QVMConnection()
import TwoQubitCompilation_organized as mod

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

    def test_convert_bloch_to_unitary(self):
        unitary = qp.rand_unitary(2)
        bloch = mod.convert_unitary_to_bloch(unitary)
        reconstruct = mod.convert_bloch_to_unitary(bloch)
        thisCorrect = self.unitary_equivalence(reconstruct,unitary)
        self.assertTrue(thisCorrect)

    def test_Cirac_unitaries(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.UsingCPhase(unitary)
        for clifford in range(6):
            unitaries = compiled.Cirac_unitaries[clifford]
            params = compiled.Cirac_parameters[clifford][-3:]
            basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
            W4 = (-0.5j*sum([x*y for x,y in zip(params,basis)])).expm()
            reconstruct = qp.tensor([unitaries[1],unitaries[0]])
            reconstruct = W4 * reconstruct
            reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * reconstruct
            thisCorrect = self.unitary_equivalence(qp.Qobj(reconstruct),compiled.unitary)
            with self.subTest(clifford = clifford):
                self.assertTrue(thisCorrect)

    def test_Cirac_parameters(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.UsingCPhase(unitary)
        for clifford in range(6):
            params = compiled.Cirac_parameters[clifford]
            basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
            W4 = (-0.5j*sum([x*y for x,y in zip(params[-3:],basis)])).expm()
            unitaries = [mod.convert_bloch_to_unitary(params[4*i:4*i+4]) for i in range(4)]
            reconstruct = qp.tensor([unitaries[1],unitaries[0]])
            reconstruct = W4 * reconstruct
            reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * reconstruct
            thisCorrect = self.unitary_equivalence(qp.Qobj(reconstruct),compiled.unitary)
            with self.subTest(clifford = clifford):
                self.assertTrue(thisCorrect)

    def test_decomposition_1Q_gates(self):
        unitary = qp.rand_unitary(2)
        implementations = mod.decomposition_1Q_gates(unitary, 0)
        for i, circuit in implementations.items():
            reconstruct = self.QVM_get_unitary_matrix(circuit, [0])
            thisCorrect = self.unitary_equivalence(qp.Qobj(reconstruct),unitary)
            with self.subTest(item = i):
                self.assertTrue(thisCorrect)

    def test_correct(self):




    @unittest.skip('Not yet ready.')
    def test_compile(self):
        unitary = qp.rand_unitary(4)
        compiled = mod.UsingCPhase(unitary)
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
