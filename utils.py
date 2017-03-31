"""
Basic functions for Quantum Information Processing
"""


import pylab as py
import qutip as qp
import itertools,os,sys,cmath

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

def getBlochForm(operator,filterResults=True,arrayForm=False,realPartOnly=False):
	"""
	Get a representation of an operator in terms of a Bloch vector
	"""
	threshold = 1e-12
	nQubits = getNumberOfQubits(operator)
	basis = getPauliBasis(nQubits)
	normalization = 2**(-nQubits)
	if not arrayForm:
		bloch = {}
		for label,op in basis.items():
			expect = qp.expect(op,operator)
			if realPartOnly:
				expect = expect.real
			if (not filterResults) or (abs(expect) > threshold):
				bloch[label] = expect * normalization
		return bloch
	else:
		# Return in a vector format
		array = []
		keys = bloch.keys()
		keys.sort()
		for key in keys:
			expect = qp.expect(basis[key],operator)
			if realPartOnly:
				expect = expect.real
			array.append(expect * normalization)
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

def getTraceDistance(operatorA,operatorB,unitaryPhaseCorrection=False):
	"""
	Get the trace distance between operators
	"""
	if unitaryPhaseCorrection:
		return qp.tracedist(getUnitaryToNormalForm(operatorA),getUnitaryToNormalForm(operatorB))
	else:
		return qp.tracedist(operatorA,operatorB)

def getNumberOfQubits(operator):
	"""
	Return the number of qubits in the Hilbert space where the operator is defined
	"""
	if not isSystemOfQubits(operator):
		raise ValueError('Operator must be defined on a collection of qubits')
	nQubits = len(operator.dims[0])
	return nQubits


def convertUnitaryToNormalForm(unitary):
	"""
	Multiply an unitary by a phase factor so that it has real trace.
	If trace is zero, consider the Pauli expansion.
	"""
	threshold = 1e-12
	nQubits = getNumberOfQubits(unitary)
	for lab,op in getPauliBasis(nQubits).items():
		trace = (op * unitary).tr()
		if abs(trace) > threshold:
			phase = cmath.phase(trace)
			break
	return py.exp(-1j*phase) * unitary


def getSingleQubitRotation(angle,axis):
	"""
	Return an unitary operator defined on a single qubit
	"""
	ketZ = [qp.basis(2,x) for x in range(2)]
	if axis.lower() in ['z','+z']:
		kets = ketZ
	elif axis.lower() in ['x','+x']:
		kets = [(ketZ[0]+x*ketZ[1]).unit() for x in [+1,-1]]
	elif axis.lower() in ['y','+y']:
		kets = [(ketZ[0]+x*ketZ[1]).unit() for x in [+1j,-1j]]
	elif axis.lower() in ['-z']:
		kets = ketZ[::-1]
	elif axis.lower() in ['-x']:
		kets = [(ketZ[0]+x*ketZ[1]).unit() for x in [-1,+1]]
	elif axis.lower() in ['-y']:
		kets = [(ketZ[0]+x*ketZ[1]).unit() for x in [-1j,+1j]]
	unitary = qp.ket2dm(kets[0]) + py.exp(-1j*angle) * qp.ket2dm(kets[1])
	return unitary

def getUnitaryRotation(angle,axis):
	"""
	Return an unitary operator defined on multiple qubits
	"""
	unitaries = [getSingleQubitRotation(x,y) for x,y in zip(angle,axis)]
	unitary = reduce(lambda old,new: new * old,unitaries)
	return unitary

def getControlledPauli(nQubits=2,bits=[0,1],basis='zz'):
	"""
	Return the controlled gate in Pauli basis between bits
	"""
	idList = [qp.qeye(2)] * nQubits
	sigA = _pauliBasis[1][basis[0].upper()]
	sigB = _pauliBasis[1][basis[1].upper()]

	II = qp.tensor(idList)
	opList = idList[:]
	opList[bits[0]] = sigA
	SI = qp.tensor(opList)
	opList = idList[:]
	opList[bits[1]] = sigB
	IS = qp.tensor(opList)
	opList = idList[:]
	opList[bits[0]] = sigA
	opList[bits[1]] = sigB
	SS = qp.tensor(opList)
	operator = (II + SI + IS - SS) / 2.0
	return operator

def getHadamardGate(nQubits=1):
	"""
	Return the Hadamard gate on multiple qubits
	"""
	return qp.tensor([qp.snot()] * nQubits)



def _gitcommit(message=None):
	os.system('git add ' + __file__)
	os.system('git commit -m "' + ('' if message is None else str(message)) + '"')
	os.system('git push origin master')
