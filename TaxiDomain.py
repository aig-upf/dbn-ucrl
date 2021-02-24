
import random
import numpy as np
from FactoredStruct import FactoredStruct
from FactoredMDP import FactoredMDP

class TaxiDomain( FactoredMDP ):

	def __init__( self ):
		super().__init__( 4, 2, [5, 5, 6, 4, 6] )

		# create reward structure
		scope  = np.array( [0, 1, 2, 4] )
		params = np.zeros( self.nelements( scope ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if ( state[3] == 4 and
			     not np.array_equal( state, [0, 4, 0, 4] ) and
			     not np.array_equal( state, [3, 0, 1, 4] ) and
			     not np.array_equal( state, [4, 4, 2, 4] ) and
			     not np.array_equal( state, [0, 0, 3, 4] ) ):
				params[sa] = -9

			if ( state[3] == 5 and
			     not np.array_equal( state, [0, 4, 4, 5] ) and
			     not np.array_equal( state, [3, 0, 4, 5] ) and
			     not np.array_equal( state, [4, 4, 4, 5] ) and
			     not np.array_equal( state, [0, 0, 4, 5] ) ):
				params[sa] = -9

		self.rewardstruct[0] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		self.rewardstruct[1] = FactoredStruct( np.array( [2] ), -np.ones( 6 ), np.zeros( ( self.nstates * self.nactions, 6 ) ) )
		self.rewardstruct[1].params[5] = 20

		# create transition structure

		# x-coord
		scope  = np.array( [0, 1, 2, 4] )
		params = np.zeros( ( self.nelements( scope ), 5 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[2] == 5:
				params[sa, 0] = 0.2
				params[sa, 1] = 0.2
				params[sa, 2] = 0.2
				params[sa, 3] = 0.2
				params[sa, 4] = 0.2

			elif ( state[3] == 2 and
			       not state[0] == 4 and
			       not np.array_equal( state[[0,1]], [0, 0] ) and
			       not np.array_equal( state[[0,1]], [0, 1] ) and
			       not np.array_equal( state[[0,1]], [1, 3] ) and
			       not np.array_equal( state[[0,1]], [1, 4] ) and
			       not np.array_equal( state[[0,1]], [2, 0] ) and
			       not np.array_equal( state[[0,1]], [2, 1] ) ):
				params[sa, state[0]] = 0.2
				params[sa, state[0] + 1] = 0.8

			elif ( state[3] == 3 and
			       not state[0] == 0 and
			       not np.array_equal( state[[0,1]], [1, 0] ) and
			       not np.array_equal( state[[0,1]], [1, 1] ) and
			       not np.array_equal( state[[0,1]], [2, 3] ) and
			       not np.array_equal( state[[0,1]], [2, 4] ) and
			       not np.array_equal( state[[0,1]], [3, 0] ) and
			       not np.array_equal( state[[0,1]], [3, 1] ) ):
				params[sa, state[0]] = 0.2
				params[sa, state[0] - 1] = 0.8

			else:
				params[sa, state[0]] = 1;

		self.transitionstruct[0] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# y-coord
		scope  = np.array( [1, 2, 4] )
		params = np.zeros( ( self.nelements( scope ), 5 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[1] == 5:
				params[sa, 0] = 0.2
				params[sa, 1] = 0.2
				params[sa, 2] = 0.2
				params[sa, 3] = 0.2
				params[sa, 4] = 0.2

			elif state[2] == 0 and state[0] < 4:
				params[sa, state[0]] = 0.2
				params[sa, state[0] + 1] = 0.8

			elif state[2] == 1 and state[0] > 0:
				params[sa, state[0]] = 0.2
				params[sa, state[0] - 1] = 0.8

			else:
				params[sa, state[0]] = 1

		self.transitionstruct[1] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# passenger
		scope  = np.array( [0, 1, 2, 3, 4] )
		params = np.zeros( ( self.nelements( scope ), 6 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[2] == 5:
				params[sa, 0] = 0.25
				params[sa, 1] = 0.25
				params[sa, 2] = 0.25
				params[sa, 3] = 0.25

			elif ( np.array_equal( state[[0, 1, 2, 4]], [0, 4, 0, 4] ) or
			       np.array_equal( state[[0, 1, 2, 4]], [3, 0, 1, 4] ) or
			       np.array_equal( state[[0, 1, 2, 4]], [4, 4, 2, 4] ) or
			       np.array_equal( state[[0, 1, 2, 4]], [0, 0, 3, 4] ) ):
				params[sa, 4] = 1

			elif np.array_equal( state[[0, 1, 2, 4]], [0, 4, 4, 5] ):
				if state[3] == 0:
					params[sa, 5] = 1
				else:
					params[sa, 0] = 1

			elif np.array_equal( state[[0, 1, 2, 4]], [3, 0, 4, 5] ):
				if state[3] == 1:
					params[sa, 5] = 1
				else:
					params[sa, 1] = 1

			elif np.array_equal( state[[0, 1, 2, 4]], [4, 4, 4, 5] ):
				if state[3] == 2:
					params[sa, 5] = 1
				else:
					params[sa, 2] = 1

			elif np.array_equal( state[[0, 1, 2, 4]], [0, 0, 4, 5] ):
				if state[3] == 3:
					params[sa, 5] = 1
				else:
					params[sa, 3] = 1

			else:
				params[sa, state[2]] = 1

		self.transitionstruct[2] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		# destination
		scope  = np.array( [2, 3] )
		params = np.zeros( ( self.nelements( scope ), 4 ) )
		for sa in range( np.size( params, 0 ) ):
			state = self.decode( sa, scope )

			if state[0] == 5:
				params[sa, 0] = 0.25
				params[sa, 1] = 0.25
				params[sa, 2] = 0.25
				params[sa, 3] = 0.25

			else:
				params[sa, state[1]] = 1;

		self.transitionstruct[3] = FactoredStruct( scope, params, np.zeros( ( self.nstates * self.nactions, np.size( params, 0 ) ) ) )

		self.assignmappings()

	def resetstate( self ):
		state = [0] * 4
		state[0] = random.randint( 0, 4 )
		state[1] = random.randint( 0, 4 )
		state[2] = random.randint( 0, 3 )
		state[3] = random.randint( 0, 3 )

		return self.encode( np.array( state ), range( self.nstatefactors ) )

