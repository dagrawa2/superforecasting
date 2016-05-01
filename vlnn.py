from __future__ import division
import numpy as np
import networkx as nx


# Let x be a real number.
# UnitInterval(x) forces x to stay in [0, 1].

def UnitInterval(x):
	if x < 0:
		return 0
	elif x > 1:
		return 1
	else:
		return x

# Let A be a 1D array of the form
# A = np.array([#, #]).
# UnitSquare(A) applies UnitInterval to A elementwise.

def UnitSquare(A):
	for i in range(2):
		A[i] = UnitInterval(A[i])
	return A

# Let L and M be lists without redundancies.
# ListIntersection(L, M) returns the list of elements common to L and M.

def ListIntersection(L, M):
	return [l for l in L if l in M]

# Let ListOfLists be a list of disjoint lists,
# where each list is without redundancies.
# ListUnion(ListOfLists) returns a list that is a union of all input lists.
# The output contains no redundancies,
# and it preserves the order of elements as they appear in the list of lists.

def ListUnion(ListOfLists):
	L = []
	for M in ListOfLists:
		L = L + M
	return L


class NeuralNet(nx.DiGraph):

# self.__init__(order, s) defines an instance of a DiGraph.
# "order" is the number of nodes.
# "s" is a parameter that defines the steepness of the function self.Sigma (see below).

	def __init__(self, order, s):
		nx.DiGraph.__init__(self)
		self.Order = order
		self.s = s
		self.add_nodes_from(range(order))
		self.y = [np.array([0, 0]) for j in range(order)]
		self.dLdy = [0 for j in range(order)]
		self.SubgraphLayers = [[[j]] for j in range(order)]
		self.Layers = []
		self.w = {}
		self.dLdw = {}

# Let N be a leaf node.
# SetLeaf(N, x) defines P(N=1) = x.

	def SetLeaf(self, N, x):
		self.y[N] = np.array([1-x, x])

# Let M and N be nodes.
# AddArc(M, N) adds an arc from M to N and defines
# P(N=1 | M=0) = w_0 and P(N=1 | M=1) = w_1.

	def AddArc(self, M, N, w_0, w_1):
		self.add_edge(M, N)
		self.w.update({(M,N):np.array([w_0, w_1])})
		self.dLdw.update({(M,N):np.array([0,0])})

# Let M and N be nodes.
# SuccessorsInSubgraph(N, M) returns the list of successors of N that are contained in the subgraph formed by M and its ancestors.

	def SuccessorsInSubgraph(self, N, M):
		return ListIntersection(self.successors(N), [M]+list(nx.ancestors(self, M)))

# Sigma is a continuous sigmoid function mapping [0, 1] onto [0, 1].
# Sigma has endpoints (0, 0) and (1, 1).
# Sigma passes through and is symmetric about (1/2, 1/2).
# The parameter self.s is the tangential slope of Sigma at (1/2, 1/2).
# Sigma is tangentially flat at (0, 0) and (1, 1).

	def Sigma(self, x):
		if x == 0:
			return 0
		elif x == 1:
			return 1
		else:
			return (1+np.tanh(self.s*np.arctanh(2*x-1)))/2

# Tau is the derivative of Sigma as a function of the output of Sigma:
# Tau(y) = Sigma'(Sigma^{-1}(y)).

	def Tau(self, y):
		if y == 0 or y == 1:
			return 0
		else:
			x = (1+np.tanh(1/self.s*np.arctanh(2*y-1)))/2
			return self.s*(1-(2*y-1)**2)/(1-(2*x-1)**2)

# CreateLayers() organizes the nodes of self into layers such that
# (1) all leaves are in the bottom-most layer and
# (2) all arcs connect nodes to nodes in higher layers,
# where the root node is in the top-most (highest) layer.
# Layers() also defines layers for the subgraph formed by a given node and its ancestors;
# the subgraph layers are intersections of the layers of self with the subgraph.

	def CreateLayers(self):
		H = nx.DiGraph(self)
		for i in range(self.Order):
			L = [N for N in H.nodes() if H.in_degree(N) == 0]
			self.Layers.append(L)
			H.remove_nodes_from(L)
			if H.nodes() == []:
				break
		self.Layers.reverse()
		for l in range(len(self.Layers)-1):
			for M in self.Layers[l]:
				for i in range(l+1, len(self.Layers)):
					self.SubgraphLayers[M].append(ListIntersection(self.Layers[i], list(nx.ancestors(self, M))))

# Let A be the root node of self.
# Run() computes the probability of every node,
# and it returns P(A=1).
# If ShowOutput is set to False,
# then no output is returned.

	def Run(self, ShowOutput=True):
		NodesInReverse = ListUnion(self.Layers[:-1])
		NodesInReverse.reverse()
		for M in NodesInReverse:
			Temp = self.Sigma(np.mean([self.w[(N,M)].dot(self.y[N]) for N in self.predecessors(M)]))
			self.y[M] = np.array([1-Temp, Temp])
		if ShowOutput:
			return self.y[0][1]

# Let M be a node.
# Run() Computes P(M=1) along every arc pointing to M
# and then takes an average.
# StandardDev(M) returns the standard deviation in the above computation.

	def StandardDev(self, M):
		self.Run(ShowOutput=False)
		return np.sqrt(np.mean([(self.w[(N,M)].dot(self.y[N])-np.mean([self.w[(O,M)].dot(self.y[O]) for O in self.predecessors(M)]))**2 for N in self.predecessors(M)]))

# Define the sum of variances
# V = sum_{M in Nodes} StandardDev(M)^2.
# DecreaseVariance(alpha) uses gradient descent to adjust the weights on the arcs and the initial values on the leaves such that
# the total variance V decreases.
# "alpha" is the step size used for the gradient descent.

	def DecreaseVariance(self, alpha):
		self.Run(ShowOutput=False)
		for M in range(self.Order):
			self.dLdy[M] = 0
		for M in ListUnion(self.Layers[:-1]):
			for N in self.SubgraphLayers[M][1]:
				self.dLdy[N] += (self.w[(N,M)][1]-self.w[(N,M)][0])*(self.w[(N,M)].dot(self.y[N]) - np.mean([self.w[(O,M)].dot(self.y[O]) for O in self.predecessors(M)]))/self.in_degree(M)
			if len(self.SubgraphLayers[M]) >= 3:
				for N in ListUnion(self.SubgraphLayers[M][2:]):
					self.dLdy[N] += sum([(self.w[(N,O)][1]-self.w[(N,O)][0])*self.Tau(self.y[O][1])*self.dLdy[O]/self.in_degree(O) for O in self.SuccessorsInSubgraph(N, M)])
		for M in self.Layers[-1]:
			Temp = UnitInterval(self.y[M][1] - alpha*self.dLdy[M])
			self.y[M] = np.array([1-Temp, Temp])
		for N in self.predecessors(0):
			self.dLdw[(N,0)] = self.y[N]*(self.w[(N,0)].dot(self.y[N]) - np.mean([self.w[(O,0)].dot(self.y[O]) for O in self.predecessors(0)]))/self.in_degree(0)
		for M in ListUnion(self.Layers[1:-1]):
			for N in self.predecessors(M):
				self.dLdw[(N,M)] = self.y[N]*self.Tau(self.y[M][1])*self.dLdy[M]/self.in_degree(M)
		for E in self.edges():
			self.w[E] = UnitSquare(self.w[E] - alpha*self.dLdw[E])

# Let A be the root node.
# Let y=1 if a real instance of event A happens,
# and let y=0 if event A does not happen.
# Train(y, alpha) uses gradient descent to adjust the weights on the arcs and initial values on the leaves such that
# the squared error between y and the prediction of P(A=1) made by Run() decreases.
# "alpha" is the step size used for gradient descent.

	def Train(self, y, alpha):
		self.Run(ShowOutput=False)
		self.dLdy[0] = self.y[0][1]-y
		for N in ListUnion(self.Layers[1:]):
			for O in self.successors(N):
				self.dLdw[(N,O)] = self.y[N]*self.Tau(self.y[O][1])*self.dLdy[O]/self.in_degree(O)
			self.dLdy[N] = sum([(self.w[(N,O)][1]-self.w[(N,O)][0])*self.Tau(self.y[O][1])*self.dLdy[O]/self.in_degree(O) for O in self.successors(N)])
		for E in self.edges():
			self.w[E] = UnitSquare(self.w[E] - alpha*self.dLdw[E])
		for M in self.Layers[-1]:
			Temp = UnitInterval(self.y[M][1] - alpha*self.dLdy[M])
			self.y[M] = np.array([1-Temp, Temp])


# Example:

# Define a NeuralNet G with 6 nodes and s=2.

G = NeuralNet(6, 2)
G.AddArc(1, 0, 0.5, 0.7)
G.AddArc(2, 0, 0.5, 0.8)
G.AddArc(3, 1, 0.3, 0.6)
G.AddArc(4, 1, 0.7, 0.5)
G.AddArc(4, 2, 0.2, 0.9)
G.AddArc(5, 2, 0.8, 0.1)
G.SetLeaf(3, 0.7)
G.SetLeaf(4, 0.8)
G.SetLeaf(5, 0.2)
G.CreateLayers()

# Iterate DecreaseVariance 200 times with step size alpha=0.05.
# After every 50 iterations,
# Print StandardDev(M) averaged over all nonleaf nodes M.
# The average StandardDev(M) should decrease with the number of iterations.

for i in range(5):
	for j in range(50*i):
		G.DecreaseVariance(0.05)
	print np.mean([G.StandardDev(M) for M in range(3)])