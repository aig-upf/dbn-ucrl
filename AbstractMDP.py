
import random
import numpy as np

class AbstractMDP:

	def __init__( self, nstates, nactions ):
		self.nstates  = nstates
		self.nactions = nactions

	def act( self, state, action ):
		pass

	def resetstate( self ):
		pass

	def greedy( self, Q ):
		Qs = np.reshape( Q , [ self.nstates, self.nactions ] )
		pi = np.argmax ( Qs, 1 )
		V  = np.choose ( pi, Qs.T )
		return V, pi

