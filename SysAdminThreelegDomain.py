
import numpy as np
from FactoredStruct import FactoredStruct
from FactoredMDP import FactoredMDP

class SysAdminThreelegDomain( FactoredMDP ):
	
	def __init__( self, legcomps ):
		self.ncomps = 3 * legcomps + 1
		super().__init__( self.ncomps, self.ncomps, [2] * self.ncomps + [self.ncomps + 1] )

		# create reward structure
		for i in range( self.ncomps ):
			self.rewardstruct[i] = FactoredStruct( np.array( [i] ), np.arange( 2 ), np.zeros( ( self.nstates * self.nactions, 2 ) ) )

		# create transition structure
		scope  = np.array( [0, self.ncomps] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[1] == 0:
				params[sa, 0] = 0.05;
				params[sa, 1] = 0.95;
			elif state[0] == 1:
				params[sa, 0] = 0.10;
				params[sa, 1] = 0.90;
			else:
				params[sa, 0] = 0.99;
				params[sa, 1] = 0.01;

		self.transitionstruct[0] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		for i in range( 1, self.ncomps ):
			prev   = max( 0, i - 3 )
			scope  = np.array( [prev, i, self.ncomps] )
			params = np.zeros( ( self.nelements( scope ), 2 ) )
			for sa in range( np.size( params, 0 ) ):
				state = self.decode( sa, scope )

				if state[2] == i:
					params[sa, 0] = 0.05
					params[sa, 1] = 0.95
				elif state[1] == 0:
					params[sa, 0] = 0.99
					params[sa, 1] = 0.01
				elif state[0] == 1:
					params[sa, 0] = 0.10
					params[sa, 1] = 0.90
				else:
					params[sa, 0] = 0.33
					params[sa, 1] = 0.67

			self.transitionstruct[i] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		self.assignmappings()

	def resetstate( self ):
		return self.encode( np.array( [0] * self.ncomps ), range( self.nstatefactors ) )

