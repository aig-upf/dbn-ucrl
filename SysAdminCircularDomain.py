
import numpy as np
from FactoredStruct import FactoredStruct
from FactoredMDP import FactoredMDP

class SysAdminCircularDomain( FactoredMDP ):
	
	def __init__( self, ncomps ):
		super().__init__( ncomps, ncomps, [2] * ncomps + [ncomps + 1] )

		self.ncomps = ncomps

		# create reward structure
		for i in range( ncomps ):
			self.rewardstruct[i] = FactoredStruct( np.array( [i] ), np.arange( 2 ), np.zeros( ( self.nstates * self.nactions, 2 ) ) )

		# create transition structure
		for i in range( ncomps ):
			prev   = ( ncomps + i - 1 ) % ncomps
			scope  = np.array( [prev, i, ncomps] )
			params = np.zeros( ( self.nelements( scope ), 2 ) )
			for sa in range( np.size( params, 0 ) ):
				state = self.decode( sa, scope )

				if state[2] == i:
					params[sa, 1] = 1
				elif state[0] == 0 and state[1] == 0:
					params[sa, 0] = 0.9762
					params[sa, 1] = 0.0238
				elif state[0] == 0 and state[1] == 1:
					params[sa, 0] = 0.525
					params[sa, 1] = 0.475
				elif state[0] == 1 and state[1] == 0:
					params[sa, 0] = 0.9525
					params[sa, 1] = 0.0475
				else:
					params[sa, 0] = 0.05
					params[sa, 1] = 0.95

			self.transitionstruct[i] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		self.assignmappings()

	def resetstate( self ):
		return self.encode( np.array( [0] * self.ncomps ), range( self.nstatefactors ) )

