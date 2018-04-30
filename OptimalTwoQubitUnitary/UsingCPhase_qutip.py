import qutip as qp
import itertools
import pylab as py
import functools

__all__ = ['two_qubit_unitary']

def rotation_z(theta):
    hamiltonian = qp.sigmaz() * theta/2
    return (-1j*hamiltonian).expm()
def rotation_x(theta):
    hamiltonian = qp.sigmax() * theta/2
    return (-1j*hamiltonian).expm()
def rotation_y(theta):
    hamiltonian = qp.sigmay() * theta/2
    return (-1j*hamiltonian).expm()
def CPhase(theta):
    phase = py.exp(1j*theta)
    return qp.Qobj(py.diag([1,1,1,phase]),dims=[[2,2],[2,2]])


def two_qubit_unitary(alpha, beta, gamma):
    """
    Implements the two-qubit unitary
        U = e^{-iH}
    where
        H = (alpha/2) * ZZ + (beta/2) * XX + (gamma/2) * YY
    in the form
        U = (W3 x W2) W4 (W1 x W0)
    """
    # implementations = []
    # for sign in itertools.product([+1,-1],repeat=6):
    #     d0 = -2*sign[0]*sign[1] * gamma
    #     d1 = -2*sign[0]*sign[1] * alpha
    #     d2 = -2*sign[2]*sign[3]*sign[4]*sign[5] * beta
    #     theta0 = alpha - sign[4] * py.pi/2
    #     theta1 = alpha - sign[5] * py.pi/2
    #
    #     angle = sign[0]*sign[1]*gamma
    #     W0 = rotation_z(angle) * rotation_x(sign[0]*py.pi/2)
    #     W1 = rotation_z(angle) * rotation_x(sign[1]*py.pi/2)
    #     angleA = sign[2]*sign[3]*sign[4]*sign[5]*beta + sign[4]*py.pi/2
    #     angleB = sign[2]*sign[3]*sign[4]*sign[5]*beta + sign[5]*py.pi/2
    #     W2 = rotation_y(sign[2]*sign[4]*py.pi/2) * rotation_z(angleA)
    #     W3 = rotation_y(sign[3]*sign[5]*py.pi/2) * rotation_z(angleB)
    #
    #     W4unitaries = []
    #     W4unitaries.append(CPhase(d0))
    #     W4unitaries.append(qp.tensor([rotation_x(s*py.pi/2) for s in [sign[1],sign[0]]]))
    #     W4unitaries.append(qp.tensor([rotation_z(alpha-py.pi/2+s*py.pi) for s in [sign[5],sign[4]]]))
    #     W4unitaries.append(CPhase(d1))
    #     W4unitaries.append(qp.tensor([rotation_x(s*py.pi/2) for s in [sign[3],sign[2]]]))
    #     W4unitaries.append(CPhase(d2))
    #     W4 = functools.reduce(lambda old,new : new*old, W4unitaries)
    #     implementations.append((W0,W1,W2,W3,W4))
    #
    # # mock = [qp.sigmaz()]*4 + [qp.tensor([qp.sigmax()]*2)]
    # # return [mock]*64
    # return implementations
    implementations = []

    for kz in itertools.product([+1,-1],repeat=2):
        for kx in itertools.product([+1,-1],repeat=2):
            for ky in itertools.product([+1,-1],repeat=2):

                gamma_corr = ky[0]*ky[1]*gamma
                alpha_corr = alpha
                beta_corr = kx[0]*kx[1]*kz[0]*kz[1]*beta
                angle_x1 = lambda s: beta_corr + py.pi/2 if s>0 else beta_corr - py.pi/2
                angle_z1 = lambda s: alpha_corr - py.pi/2 if s>0 else alpha_corr + py.pi/2

                W0 = rotation_z(gamma_corr) * rotation_x(-ky[0]*py.pi/2)
                W1 = rotation_z(gamma_corr) * rotation_x(-ky[1]*py.pi/2)
                W2 = rotation_y(-kx[0]*kz[0]*py.pi/2) * rotation_z(angle_x1(kz[0]))
                W3 = rotation_y(-kx[1]*kz[1]*py.pi/2) * rotation_z(angle_x1(kz[1]))

                unitaries = []
                unitaries.append(CPhase(-2*gamma_corr))
                unitaries.append(qp.tensor([rotation_x(+x*py.pi/2) for x in ky[1::-1]]))
                unitaries.append(qp.tensor([rotation_z(angle_z1(x)) for x in kz[1::-1]]))
                unitaries.append(CPhase(-2*alpha_corr))
                unitaries.append(qp.tensor([rotation_x(+x*py.pi/2) for x in kx[1::-1]]))
                unitaries.append(CPhase(-2*beta_corr))
                W4 = functools.reduce(lambda old,new : new*old, unitaries)

                implementations.append((W0,W1,W2,W3,W4))
    return implementations




def putting_things_together(alpha, beta, gamma):
    implementations = []

    for kz in itertools.product([+1,-1],repeat=2):
        for kx in itertools.product([+1,-1],repeat=4):
            for ky in itertools.product([+1,-1],repeat=4):
                unitaries = []

                gamma_corr = ky[0]*ky[1]*gamma
                angle_y1 = lambda s: gamma_corr if s>0 else gamma_corr + py.pi
                angle_y2 = lambda s: 0 if s>0 else py.pi
                alpha_corr = alpha
                angle_z1 = lambda s: alpha_corr if s>0 else alpha_corr + py.pi
                angle_z2 = lambda s: 0 if s>0 else py.pi
                beta_corr = kx[0]*kx[1]*beta
                angle_x1 = lambda s: beta_corr if s>0 else beta_corr + py.pi
                angle_x2 = lambda s: 0 if s>0 else py.pi


                # Block for YY
                unitaries.append(qp.tensor([rotation_x(+x*py.pi/2) for x in ky[1::-1]]))
                unitaries.append(qp.tensor([rotation_z(angle_y1(x)) for x in ky[3:1:-1]]))
                unitaries.append(CPhase(-2*gamma_corr))
                unitaries.append(qp.tensor([rotation_x(-x*y*py.pi/2) for x,y in zip(ky[1::-1],ky[3:1:-1])]))
                unitaries.append(qp.tensor([rotation_z(angle_y2(x)) for x in ky[3:1:-1]]))

                # Block for ZZ
                unitaries.append(qp.tensor([rotation_z(angle_z1(x)) for x in kz[1::-1]]))
                unitaries.append(CPhase(-2*alpha_corr))
                unitaries.append(qp.tensor([rotation_z(angle_z2(x)) for x in kz[1::-1]]))

                # Block for XX
                unitaries.append(qp.tensor([rotation_y(+x*py.pi/2) for x in kx[1::-1]]))
                unitaries.append(qp.tensor([rotation_z(angle_x1(x)) for x in kx[3:1:-1]]))
                unitaries.append(CPhase(-2*beta_corr))
                unitaries.append(qp.tensor([rotation_y(-x*y*py.pi/2) for x,y in zip(kx[1::-1],kx[3:1:-1])]))
                unitaries.append(qp.tensor([rotation_z(angle_x2(x)) for x in kx[3:1:-1]]))

                unitary = functools.reduce(lambda old,new : new*old, unitaries)

                implementations.append(unitary)
    return implementations



def moving_things_around(alpha, beta, gamma):
    implementations = []

    for kz in itertools.product([+1,-1],repeat=2):
        for kx in itertools.product([+1,-1],repeat=2):
            for ky in itertools.product([+1,-1],repeat=2):
                unitaries = []

                gamma_corr = ky[0]*ky[1]*gamma
                angle_y1 = lambda s: gamma_corr if s>0 else gamma_corr + py.pi
                angle_y2 = lambda s: 0 if s>0 else py.pi
                alpha_corr = alpha
                angle_z1 = lambda s: alpha_corr - py.pi/2 if s>0 else alpha_corr + py.pi/2
                angle_z2 = lambda s: 0 if s>0 else py.pi
                beta_corr = kx[0]*kx[1]*beta
                angle_x1 = lambda s: beta_corr + py.pi/2 if s>0 else beta_corr - py.pi/2
                angle_x2 = lambda s: 0 if s>0 else py.pi

                unitaries.append(qp.tensor([rotation_x(+x*py.pi/2) for x in ky[1::-1]]))
                unitaries.append(qp.tensor([rotation_z(angle_y1(+1)) for x in range(2)]))

                unitaries.append(CPhase(-2*gamma_corr))
                unitaries.append(qp.tensor([rotation_x(-x*py.pi/2) for x in ky[1::-1]]))
                unitaries.append(qp.tensor([rotation_z(angle_z1(x)) for x in kz[::-1]]))
                unitaries.append(CPhase(-2*alpha_corr))
                unitaries.append(qp.tensor([rotation_x(+x*y*py.pi/2) for x,y in zip(kx[1::-1],kz[::-1])]))
                unitaries.append(CPhase(-2*beta_corr))

                unitaries.append(qp.tensor([rotation_z(angle_x1(x)) for x in kz[::-1]]))
                unitaries.append(qp.tensor([rotation_y(-x*py.pi/2) for x in kx[1::-1]]))

                unitary = functools.reduce(lambda old,new : new*old, unitaries)

                implementations.append(unitary)
    return implementations

import unittest
import cmath
import random
class Test_two_qubit_unpyitary(unittest.TestCase):
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

    def test_exp_ZZ(self):
        alpha = 2*py.pi*random.random()
        target = (-1j*alpha/2 * qp.tensor([qp.sigmaz()]*2)).expm()
        for counter,k in enumerate(itertools.product([+1,-1],repeat=2)):
            unitaries = []
            alpha_corr = alpha
            angle = lambda s: alpha_corr if s>0 else alpha_corr + py.pi
            unitaries.append(qp.tensor([rotation_z(angle(x)) for x in k[::-1]]))
            unitaries.append(CPhase(-2*alpha_corr))
            angle = lambda s: 0 if s>0 else py.pi
            unitaries.append(qp.tensor([rotation_z(angle(x)) for x in k[::-1]]))
            # if k[0] < 0:
            #     unitaries.append(qp.tensor([qp.qeye(2),qp.sigmaz()]))
            # if k[1] < 0:
            #     unitaries.append(qp.tensor([qp.sigmaz(),qp.qeye(2)]))
            unitary = functools.reduce(lambda old,new : new*old, unitaries)
            thisCorrect = self.unitary_equivalence(unitary,target)
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)

    def test_exp_YY(self):
        alpha = 2*py.pi*random.random()
        # targets = [(-1j*x*alpha/2 * qp.tensor([qp.sigmay()]*2)).expm() for x in [+1,-1]]
        target = (-1j*alpha/2 * qp.tensor([qp.sigmay()]*2)).expm()
        for counter,p in enumerate(itertools.product([+1,-1],repeat=4)):
            k,m = p[0:2],p[2:4]
            alpha_corr = k[0]*k[1]*alpha
            unitaries = []
            unitaries.append(qp.tensor([rotation_x(x*py.pi/2) for x in k[::-1]]))
            angle = lambda s: alpha_corr if s>0 else alpha_corr + py.pi
            unitaries.append(qp.tensor([rotation_z(angle(x)) for x in m[::-1]]))
            unitaries.append(CPhase(-2*alpha_corr))
            unitaries.append(qp.tensor([rotation_x(-x*y*py.pi/2) for x,y in zip(k[::-1],m[::-1])]))
            angle = lambda s: 0 if s>0 else py.pi
            unitaries.append(qp.tensor([rotation_z(angle(x)) for x in m[::-1]]))
            unitary = functools.reduce(lambda old,new : new*old, unitaries)
            thisCorrect = self.unitary_equivalence(unitary,target)
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)

    def test_exp_XX(self):
        alpha = 2*py.pi*random.random()
        target = (-1j*alpha/2 * qp.tensor([qp.sigmax()]*2)).expm()
        for counter,p in enumerate(itertools.product([+1,-1],repeat=4)):
            k,m = p[0:2],p[2:4]
            alpha_corr = k[0]*k[1]*alpha
            unitaries = []
            unitaries.append(qp.tensor([rotation_y(x*py.pi/2) for x in k[::-1]]))
            angle = lambda s: alpha_corr if s>0 else alpha_corr + py.pi
            unitaries.append(qp.tensor([rotation_z(angle(x)) for x in m[::-1]]))
            unitaries.append(CPhase(-2*alpha_corr))
            unitaries.append(qp.tensor([rotation_y(-x*y*py.pi/2) for x,y in zip(k[::-1],m[::-1])]))
            angle = lambda s: 0 if s>0 else py.pi
            unitaries.append(qp.tensor([rotation_z(angle(x)) for x in m[::-1]]))
            unitary = functools.reduce(lambda old,new : new*old, unitaries)
            thisCorrect = self.unitary_equivalence(unitary,target)
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)



    # @unittest.skip('Not yet there.')
    def test_unitary_decomposition(self):
        params = [2*py.pi*random.random() for x in range(3)]
        unitary = sum([-0.5j*x*y for x,y in zip(params,self.basis)]).expm()
        # isCorrect = True
        implementations = two_qubit_unitary(*params)
        self.assertEqual(len(implementations),64)
        for counter in range(len(implementations)):
            impl = implementations[counter]
            reconstruct = qp.tensor([impl[1],impl[0]])
            reconstruct = impl[4] * reconstruct
            reconstruct = qp.tensor([impl[3],impl[2]]) * reconstruct
            thisCorrect = self.unitary_equivalence(reconstruct,unitary)
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)

    @unittest.skip('Too slow.')
    def test_putting_things_together(self):
        params = [2*py.pi*random.random() for x in range(3)]
        unitary = sum([-0.5j*x*y for x,y in zip(params,self.basis)]).expm()
        # isCorrect = True
        implementations = putting_things_together(*params)
        for counter,impl in enumerate(implementations[:4]):
            thisCorrect = self.unitary_equivalence(impl,unitary)
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)

    def test_moving_things_around(self):
        params = [2*py.pi*random.random() for x in range(3)]
        unitary = sum([-0.5j*x*y for x,y in zip(params,self.basis)]).expm()
        implementations = moving_things_around(*params)
        for counter,impl in enumerate(implementations):
            thisCorrect = self.unitary_equivalence(impl,unitary)
            with self.subTest(instance = counter):
                self.assertTrue(thisCorrect)
