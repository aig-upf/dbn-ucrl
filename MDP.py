
import numpy as np
from AbstractMDP import AbstractMDP

class MDP( AbstractMDP ):

	def __init__( self, nstates, nactions, rewards, kernel ):
		super().__init__( nstates, nactions )
		self.rewards = rewards
		self.kernel  = kernel

	def act( self, state, action ):
		sa = np.ravel_multi_index( [state, action], [self.nstates, self.nactions] )
		return np.random.choice( np.arange( self.nstates ), p = self.kernel[sa,:] ), self.rewards[sa]

	def solve( self ):
		V_old  = np.zeros ( self.nstates )
		V_diff = np.arange( self.nstates )

		while max( V_diff ) - min( V_diff ) > 1e-6:
			V, pi  = self.greedy( self.rewards + self.kernel @ V_old )
			V_diff = V - V_old
			V_old  = V

		return pi, V_diff[0]

	def resetstate( self ):
		return 0

