
import numpy as np
from FactoredStruct import FactoredStruct
from FactoredMDP import FactoredMDP

class ThreeRiverSwimDomain( FactoredMDP ):

	def __init__( self, nlocations ):
		super().__init__( 3, 1, [nlocations, nlocations, nlocations, 2, 2, 2] )

		self.nlocations = nlocations

		# create reward structure
		scope  = np.arange( 6 )
		params = np.zeros ( self.nelements( scope ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[0] == 0 and state[3] == 0:
				params[sa] += 0.05
			elif state[0] == nlocations - 1 and state[3] == 1:
				params[sa] += 1

			if state[1] == 0 and state[4] == 0:
				params[sa] += 0.05
			elif state[1] == nlocations - 1 and state[4] == 1:
				params[sa] += 1

			if state[2] == 0 and state[5] == 0:
				params[sa] += 0.05
			elif state[2] == nlocations - 1 and state[5] == 1:
				params[sa] += 1

			if np.array_equal( state, [nlocations - 1, nlocations - 1, nlocations - 1, 1, 1, 1] ):
				params[sa] += 3

		self.rewardstruct[0] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )
		# normalize in [0,1]
		self.rewardstruct[0].params = self.rewardstruct[0].params / 6

		scope  = np.array( [0, 3] )
		params = np.zeros ( ( self.nelements( scope ), nlocations ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[1] == 0:
				j = max( 0, state[0] - 1 )
				params[sa, j] = 1
			elif state[0] == 0:
				params[sa, 0] = 0.4
				params[sa, 1] = 0.6
			elif state[0] == nlocations - 1:
				params[sa, nlocations - 2] = 0.4
				params[sa, nlocations - 1] = 0.6
			else:
				params[sa, state[0] - 1] = 0.05
				params[sa, state[0]    ] = 0.6
				params[sa, state[0] + 1] = 0.35

		self.transitionstruct[0] = FactoredStruct( np.array( [0, 3] ), params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )
		self.transitionstruct[1] = FactoredStruct( np.array( [1, 4] ), params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )
		self.transitionstruct[2] = FactoredStruct( np.array( [2, 5] ), params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		self.assignmappings()

	def resetstate( self ):
		return self.encode( np.array( [0, 0, 0] ), range( self.nstatefactors ) )

