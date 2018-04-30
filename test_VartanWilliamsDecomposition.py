import unittest
from VartanWilliamsDecomposition import TwoQubitOperation
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

    def test_Cirac(self, n_tests = 1):
        for i in range(n_tests):
            obj = TwoQubitOperation()
            unitary = qp.rand_unitary(4)
            obj.unitary = qp.Qobj(unitary, dims=[[2,2],[2,2]])
            obj.decomposition_Cirac()
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

    # @unittest.skip('Not ready yet')
    def test_MBQC(self):
        cz = qp.Qobj(py.diag([1,1,1,-1]),dims=[[2,2],[2,2]])
        rotx = lambda x: qp.qeye(2)*py.cos(x/2) - 1j*qp.sigmax()*py.sin(x/2)
        angles = [x*py.pi/4 for x in range(8)]
        for angle_top in angles:
            for angle_bot in angles:
                obj = TwoQubitOperation()
                unitary = cz * qp.tensor([rotx(angle_top),rotx(angle_bot)]) * cz
                obj.unitary = qp.Qobj(unitary, dims=[[2,2],[2,2]])
                obj.decomposition_Cirac()
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


if __name__ == '__main__':
    unittest.main()