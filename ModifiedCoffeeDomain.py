
import random
import numpy as np
from FactoredStruct import FactoredStruct
from FactoredMDP import FactoredMDP

class ModifiedCoffeeDomain( FactoredMDP ):

	def __init__( self ):
		super().__init__( 6, 2, [2] * 6 + [4] )

		# create reward structure
		self.rewardstruct[0] = FactoredStruct( np.array( [0] ), np.zeros( 2 ), np.zeros( ( self.nstates * self.nactions, 2 ) ) )
		self.rewardstruct[1] = FactoredStruct( np.array( [4] ), np.zeros( 2 ), np.zeros( ( self.nstates * self.nactions, 2 ) ) )
		self.rewardstruct[0].params[0] = 0.1
		self.rewardstruct[1].params[1] = 0.9

		# create transition structure

		# wet
		scope  = np.array( [0, 1, 2, 4, 6] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[3] == 1:
				params[sa, 0] = 1
			elif np.array_equal( state, [0, 0, 1, 0, 0] ):
				params[sa, 1] = 1
			else:
				params[sa, state[0]] = 1

		self.transitionstruct[0] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# umbrella
		scope  = np.array( [1, 3, 4, 6] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[2] == 1:
				params[sa, 0] = 1
			elif np.array_equal( state, [0, 0, 0, 3] ):
				params[sa, 0] = 0.1
				params[sa, 1] = 0.9
			else:
				params[sa, state[0]] = 1

		self.transitionstruct[1] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# raining
		scope  = np.array( [2, 4] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[1] == 1:
				params[sa, 0] = 0.5
				params[sa, 1] = 0.5
			else:
				params[sa, state[0]] = 1

		self.transitionstruct[2] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# location
		scope  = np.array( [3, 4, 6] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[1] == 1:
				params[sa, 0] = 1
			elif state[2] == 0:
				params[sa, state[0]] = 0.1
				params[sa, 1 - state[0]] = 0.9
			else:
				params[sa, state[0]] = 1

		self.transitionstruct[3] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# has coffee user
		scope  = np.array( [3, 4, 5, 6] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if np.array_equal( state, [0, 0, 1, 2] ):
				params[sa, 0] = 0.1
				params[sa, 1] = 0.9
			else:
				params[sa, 0] = 1

		self.transitionstruct[4] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# has coffee robot
		scope  = np.array( [3, 5, 6] )
		params = np.zeros( ( self.nelements( scope ), 2 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if np.array_equal( state, [1, 0, 1] ):
				params[sa, 0] = 0.1
				params[sa, 1] = 0.9
			elif np.array_equal( state, [0, 1, 2] ):
				params[sa, 0] = 1
			else:
				params[sa, state[1]] = 1

		self.transitionstruct[5] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		self.assignmappings()

	def resetstate( self ):
		state = [0] * 6
		state[2] = random.randint( 0, 1 )
		return self.encode( np.array( state ), range( self.nstatefactors ) )

