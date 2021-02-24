
import numpy as np
from FactoredStruct import FactoredStruct
from FactoredMDP import FactoredMDP

class RiverSwimDomain( FactoredMDP ):

	def __init__( self, nstates ):
		super().__init__( 1, 1, [nstates, 2] )

		# create reward structure
		self.rewardstruct[0] = FactoredStruct( np.array( [0, 1] ), np.zeros( 2 * nstates ), np.zeros( ( 2 * nstates, 2 * nstates ) ) )
		self.rewardstruct[0].params[ 0] = 0.05
		self.rewardstruct[0].params[-1] = 1

		# create transition structure
		params = np.zeros( ( 2 * nstates, nstates ) )
		for sa in range( 2 * nstates ):
			state = self.decode( sa, [0, 1] )
			if state[1] == 0:
				loc = max( 0, state[0] - 1 )
				params[sa, loc] = 1
			elif state[0] == 0:
				params[sa, 0] = 0.4
				params[sa, 1] = 0.6
			elif state[0] == nstates - 1:
				params[sa, state[0] - 1] = 0.4
				params[sa, state[0]] = 0.6
			else:
				params[sa, state[0] - 1] = 0.05
				params[sa, state[0]    ] = 0.6
				params[sa, state[0] + 1] = 0.35
		self.transitionstruct[0] = FactoredStruct( np.array( [0, 1] ), params, np.zeros( ( 2 * nstates, 2 * nstates ) ) )

		params

		self.assignmappings()

	def resetstate( self ):
		return self.encode( np.array( [0] ), range( self.nstatefactors ) )

