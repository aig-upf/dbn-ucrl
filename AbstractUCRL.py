
import sys
import numpy as np

class AbstractUCRL:

	def __init__( self, MDP, RCB, TCB, EVI ):
		self.MDP = MDP	# The MDP to be solved
		self.RCB = RCB	# List of functions for computing confidence bounds on reward
		self.TCB = TCB	# List of functions for computing confidence bounds on transition kernel
		self.EVI = EVI	# The version of Extended Value Iteration

	# compute upper confidence bounds on reward
	def rewardbound( self, S, r_hat, Nplus, N, delta, t ):
		r_upper = sys.maxsize * np.ones( r_hat.shape )

		for CB in self.RCB:
			LB, UB  = CB.confidencebound( S, r_hat, Nplus, N, delta, t )
			r_upper = np.minimum( r_upper, UB )

		return r_upper

	# compute lower and upper confidence bounds on transition kernel
	def transitionbound( self, S, p_hat, Nplus, N, delta, t ):
		p_lower = np.zeros( p_hat.shape )
		p_upper = np.ones ( p_hat.shape )

		for CB in self.TCB:
			LB, UB  = CB.confidencebound( S, p_hat, Nplus, N, delta, t )
			p_lower = np.maximum( p_lower, LB )
			p_upper = np.minimum( p_upper, UB )

		return p_lower, p_upper

	def updatepolicy( self, delta, t ):
		pass

	def updateparams( self, s, a, r, sp ):
		pass

	def runUCRL( self, delta, time_horizon, ival, g_opt ):
		compute_pi = True
		regret     = [0] * ( time_horizon // ival )
		tmp_regret = 0
		s_t        = self.MDP.resetstate()

		for t in range( time_horizon ):
			if compute_pi:
				pi      = self.updatepolicy( delta, t + 1 )
				#print('{}: policy {}'.format(t, pi))

			s_next, r_t = self.MDP.act     ( s_t, pi[s_t] )
			compute_pi  = self.updateparams( s_t, pi[s_t], r_t, s_next )
			s_t         = s_next

			tmp_regret  = tmp_regret + ( g_opt - np.mean( r_t ) )
			if t % ival == 0:
				regret[t // ival] = tmp_regret

		return regret

