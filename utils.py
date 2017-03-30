"""
Basic functions for Quantum Information Processing
"""


import pylab as py
import qutip as qp

pauliBasis = {}
pauliBasis[1] = {}
pauliBasis[1]['I'] = qp.qeye(2)
pauliBasis[1]['X'] = qp.sigmax()
pauliBasis[1]['Y'] = qp.sigmay()
pauliBasis[1]['Z'] = qp.sigmaz()

def isSystemOfQubits(operator):
	"""
	Check if a given operator is defined on a system of qubits
	"""
	try:
		dims = operator.dims
	except AttributeError:
		return False
	sets = [set(x) for x in dims]
	check = all([x==set([1]) or x==set([2]) for x in sets])
	return check

def getPauliBasis(nQubits):
	"""
	generate a Pauli basis for more than one qubit
	"""
	for i in range(1,nQubits+1):
		if i not in pauliBasis.keys():
			pauliBasis[i] = combineBasis(pauliBasis[i-1],pauliBasis[1])
	return {x:y for x,y in pauliBasis[nQubits].items()}

def combineBasis(dicBasisA,dicBasisB):
	"""
	Get an operator basis for the tensor product of two Hilbert spaces
	"""
	newBasis = {}
	for labelA,operatorA in dicBasisA.items():
		for labelB,operatorB in dicBasisB.items():
			label = labelA + labelB
			op = qp.tensor(operatorA,operatorB)
			newBasis[label] = op
	return newBasis

def getBlochForm(operator,arrayForm=False,realPartOnly=False):
	"""
	Get a representation in terms of a Bloch vector
	"""
	if not isSystemOfQubits(operator):
		raise ValueError('Operator must be defined on a collection of qubits')
	nQubits = len(operator.dims[0])
	basis = getPauliBasis(nQubits)
	bloch = {}
	normalization = 2**(-nQubits)
	for label,op in basis.items():
		expect = qp.expect(op,operator)
		if realPartOnly:
			expect = expect.real
		bloch[label] = expect * normalization
	if not arrayForm:
		return bloch
	else:
		# Return in a vector format
		keys = bloch.keys()
		keys.sort()
		array = [bloch[x] for x in keys]
		return array



























