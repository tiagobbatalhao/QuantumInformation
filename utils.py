"""
Basic functions for Quantum Information Processing
"""


import pylab as py
import qutip as qp
import itertools,os,sys

_pauliBasis = {}
_pauliBasis[1] = {}
_pauliBasis[1]['I'] = qp.qeye(2)
_pauliBasis[1]['X'] = qp.sigmax()
_pauliBasis[1]['Y'] = qp.sigmay()
_pauliBasis[1]['Z'] = qp.sigmaz()

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
	Generate a Pauli basis for more than one qubit
	"""
	for i in range(1,nQubits+1):
		if i not in _pauliBasis.keys():
			_pauliBasis[i] = combineBasis(_pauliBasis[i-1],_pauliBasis[1])
	return {x:y for x,y in _pauliBasis[nQubits].items()}

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
	Get a representation of an operator in terms of a Bloch vector
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

def getOperatorForm(bloch):
	"""
	Get an operator from its Bloch vector representation
	"""
	threshold = 1e-12
	log = py.log(len(bloch)) / py.log(4)
	if abs(log - round(log)) > threshold:
		raise ValueError('Input must have length equal to a power of 4')
	nQubits = int(round(log))
	if type(bloch) == type([]):
		labels = [''.join(x) for x in itertools.product(['I','X','Y','Z'],repeat=nQubits)]
		values = bloch
	elif type(bloch) == type({}):
		labels = [x for x,y in bloch.items()]
		values = [y for x,y in bloch.items()]
	basis = getPauliBasis(nQubits)
	operator = 0 * basis[labels[0]]
	for lab,val in zip(labels,values):
		operator += val * basis[lab]
	return operator

def getTraceDistance(operatorA,operatorB):
	"""
	Get the trace distance between operators
	"""
	return qp.tracedist(operatorA,operatorB)



def gitcommit(message=None):
	os.system('git add ' + __file__)
	os.system('git commit -m "' + ('' if message is None else str(message)) + '"')
	os.system('git push origin master')

