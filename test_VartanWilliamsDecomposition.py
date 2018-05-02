import unittest
from VartanWilliamsDecomposition import TwoQubitOperation
from VartanWilliamsDecomposition import TwoQubitOperation_Clifford
import pylab as py
import qutip as qp
import cmath


def unitary_equivalence(unitary_A, unitary_B):
    """
    Check if unitary_A and unitary_B are equivalent.
    """
    threshold = 1e-4
    thisCorrect = True
    should_be_identity = unitary_A.dag() * unitary_B
    eigs = should_be_identity.eigenenergies() * (1+0j)
    eigs*= py.exp(-1j*cmath.phase(eigs[0]))
    for eig in eigs:
        if abs(eig-1) > threshold:
            thisCorrect = False
            break
    return thisCorrect

class test_TwoQubitOperation(unittest.TestCase):

    @unittest.skip('Already done')
    def test_Cirac(self, n_tests = 100):
        for i in range(n_tests):
            obj = TwoQubitOperation()
            unitary = qp.rand_unitary(4)
            obj.unitary = qp.Qobj(unitary, dims=[[2,2],[2,2]])
            # obj.decomposition()
            # unitaries = obj._Cirac_unitaries
            # params = obj._Cirac_parameters
            # basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
            # W4 = (-0.5j*sum([x*y for x,y in zip(params,basis)])).expm()
            # reconstruct = qp.tensor([unitaries[1],unitaries[0]])
            # reconstruct = W4 * reconstruct
            # reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * reconstruct
            reconstruct = obj.reconstruct()
            thisCorrect = unitary_equivalence(qp.Qobj(reconstruct),obj.unitary)
            with self.subTest(test = i):
                self.assertTrue(thisCorrect)

    @unittest.skip('Already done')
    def test_MBQC(self):
        cz = qp.Qobj(py.diag([1,1,1,-1]),dims=[[2,2],[2,2]])
        rotx = lambda x: qp.qeye(2)*py.cos(x/2) - 1j*qp.sigmax()*py.sin(x/2)
        angles = [x*py.pi/4 for x in range(8)]
        for angle_top in angles:
            for angle_bot in angles:
                obj = TwoQubitOperation()
                unitary = cz * qp.tensor([rotx(angle_top),rotx(angle_bot)]) * cz
                obj.unitary = qp.Qobj(unitary, dims=[[2,2],[2,2]])
                # obj.decomposition()
                # unitaries = obj._Cirac_unitaries
                # params = obj._Cirac_parameters
                # basis = [qp.tensor([eval('qp.sigma'+x+'()')]*2) for x in 'zxy']
                # W4 = (-0.5j*sum([x*y for x,y in zip(params,basis)])).expm()
                # reconstruct = qp.tensor([unitaries[1],unitaries[0]])
                # reconstruct = W4 * reconstruct
                # reconstruct = qp.tensor([unitaries[3],unitaries[2]]) * reconstruct
                reconstruct = obj.reconstruct()
                thisCorrect = unitary_equivalence(qp.Qobj(reconstruct),obj.unitary)
                with self.subTest(test = (angle_top,angle_bot)):
                    self.assertTrue(thisCorrect)

    def test_Clifford(self, n_tests = 1):
        done_tests = 0
        while done_tests < n_tests:
            unitary = qp.rand_unitary(4)

            # Check that Cirac decomposition works
            cirac = TwoQubitOperation(unitary)
            reconstruct = cirac.reconstruct()
            thisCorrect = unitary_equivalence(qp.Qobj(reconstruct),cirac.unitary)
            if thisCorrect:
                obj = TwoQubitOperation_Clifford(unitary)
                correct, fail = 0, 0
                for j in range(96):
                    reconstruct = obj.reconstruct(index = j)
                    thisCorrect = unitary_equivalence(qp.Qobj(reconstruct),obj.unitary)

                    # Test ordering of first attempt
                    if thisCorrect:
                        correct += 1
                    else:
                        fail += 1
                    # print('Success: {:02d}\tFail: {:02d}'.format(correct,fail))
                    with self.subTest(test = (done_tests,j)):
                        self.assertTrue(thisCorrect)
                done_tests += 1

    def test_Clifford_order(self):
        unitary = qp.rand_unitary(4)
        obj = TwoQubitOperation_Clifford(unitary)
        parameters = obj.Clifford_parameters[0][-1]
        test = True
        print(parameters)
        test = test and (abs(parameters[0]) >= abs(parameters[1]))
        test = test and (abs(parameters[1]) >= abs(parameters[2]))
        test = test and (parameters[0] >= 0)
        test = test and (parameters[1] >= 0)
        self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()
