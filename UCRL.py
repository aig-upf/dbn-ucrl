
import numpy as np
from AbstractUCRL import AbstractUCRL

class UCRL( AbstractUCRL ):

	def __init__( self,   MDP, RCB, TCB, EVI ):
		super().__init__( MDP, RCB, TCB, EVI )

		self.N                  = np.zeros( MDP.nstates * MDP.nactions )
		self.N_previous_episode = np.zeros( MDP.nstates * MDP.nactions )
		self.v                  = np.zeros( MDP.nstates * MDP.nactions )
		self.r_hat              = np.zeros( MDP.nstates * MDP.nactions )
		self.NN                 = np.zeros( ( MDP.nstates * MDP.nactions, MDP.nstates ) )
		self.p_hat              = np.zeros( ( MDP.nstates * MDP.nactions, MDP.nstates ) )

	def updatepolicy( self, delta, t ):
		# compute confidence bounds
		Nplus                   = np.maximum( 1, self.N  )
		real_delta              = delta / np.size( Nplus )
		r_upper                 = self.rewardbound    ( self.MDP.nstates, self.r_hat, Nplus, self.N , real_delta, t )
		p_lower, p_upper        = self.transitionbound( self.MDP.nstates, self.p_hat, Nplus, self.NN, real_delta, t )

		# update counts
		self.N_previous_episode = np.copy ( self.N       )
		self.v                  = np.zeros( self.N.shape )

		# run Extended Value Iteration
		return self.EVI.computepolicy( self.MDP, r_upper, p_lower, self.p_hat, p_upper, t )

	def updateparams( self, s, a, r, sp ):
		sa = np.ravel_multi_index( [s, a], [self.MDP.nstates, self.MDP.nactions] )

		self.N[sa]        = self.N[sa] + 1
		self.v[sa]        = self.v[sa] + 1
		self.r_hat[sa]    = ( self.r_hat[sa] * ( self.N[sa] - 1 ) + r ) / self.N[sa]
		self.NN[sa, sp]   = self.NN[sa, sp] + 1
		self.p_hat[sa, :] = self.NN[sa, :] / self.N[sa]

		return self.v[sa] >= np.maximum( self.N_previous_episode[sa], 1 )

