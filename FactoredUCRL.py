
import numpy as np
from AbstractUCRL import AbstractUCRL

class FactoredUCRL( AbstractUCRL ):

	def __init__( self,   MDP, RCB, TCB, EVI ):
		super().__init__( MDP, RCB, TCB, EVI )

		# counts and sums for reward factors
		self.XR = MDP.rewardsizes();
		self.NR = [None] * MDP.nrewardfactors
		self.SR = [None] * MDP.nrewardfactors
		for i in range( MDP.nrewardfactors ):
			self.NR[i] = np.zeros( self.XR[i] )
			self.SR[i] = np.zeros( self.XR[i] )

		# counts for state factors
		self.XP = MDP.transitionsizes();
		self.NP = [None] * MDP.nstatefactors
		for i in range( MDP.nstatefactors ):
			self.NP[i] = np.zeros( ( self.XP[i], MDP.factordomains[i] ) )

		# counts that determine whether to update policy
		self.N_prev = [None] * ( MDP.nrewardfactors + MDP.nstatefactors )
		self.v      = [None] * ( MDP.nrewardfactors + MDP.nstatefactors )

	def updatepolicy( self, delta, t ):
		# compute upper bound on full reward
		r_upper = np.zeros( self.MDP.nstates * self.MDP.nactions )

		for i in range( self.MDP.nrewardfactors ):
			# compute local reward factor estimate
			Nplus      = np.maximum( self.NR[i], 1 )
			r_tot_hat  = self.SR[i] / Nplus

			# compute confidence bounds on local reward factor
			# if desired, "self.MDP.rewardstruct[i].params" allows access to true reward
			real_delta = delta / ( self.MDP.nrewardfactors * np.size( Nplus ) )
			r_tilde    = self.rewardbound( 1, r_tot_hat, Nplus, self.MDP.rewardstruct[i].params, real_delta, t )
 
			# update bound on full reward
			r_upper    = r_upper + ( self.MDP.rewardstruct[i].mapping @ r_tilde )

			# update counts
			self.N_prev[i] = np.copy ( self.NR[i] )
			self.v     [i] = np.zeros( self.XR[i] )

		# compute estimate and bounds on full transition kernel
		p_lower = np.ones ( ( self.MDP.nstates * self.MDP.nactions, self.MDP.nstates ) )
		p_hat   = np.ones ( ( self.MDP.nstates * self.MDP.nactions, self.MDP.nstates ) )
		p_upper = np.ones ( ( self.MDP.nstates * self.MDP.nactions, self.MDP.nstates ) )

		for i in range( self.MDP.nstatefactors ):
			# compute local state factor estimate
			NP_sum          = np.sum( self.NP[i], 1 )
			ix              = NP_sum > 0
			p_tot_hat       = np.ones( ( self.XP[i], self.MDP.factordomains[i] ) ) / self.MDP.factordomains[i]
			p_tot_hat[ix,:] = self.NP[i][ix,:] / NP_sum[ix,None]

			# compute confidence bounds on local state factor
			Nplus      = np.maximum( NP_sum, 1 )
			real_delta = delta / ( self.MDP.nstatefactors * np.size( Nplus ) )
			LB, UB     = self.transitionbound( self.MDP.factordomains[i], p_tot_hat, Nplus, self.NP[i], real_delta, t )

			# update bounds on full transition kernel
			p_lower = p_lower * ( self.MDP.transitionstruct[i].mapping @ LB         @ self.MDP.statemappings[i] )
			p_hat   = p_hat   * ( self.MDP.transitionstruct[i].mapping @ p_tot_hat  @ self.MDP.statemappings[i] )
			p_upper = p_upper * ( self.MDP.transitionstruct[i].mapping @ UB         @ self.MDP.statemappings[i] )

			# update counts
			self.N_prev[self.MDP.nrewardfactors + i] = np.copy ( NP_sum     )
			self.v     [self.MDP.nrewardfactors + i] = np.zeros( self.XP[i] )

		# run Extended Value Iteration
		return self.EVI.computepolicy( self.MDP, r_upper, p_lower, p_hat, p_upper, t )

	def updateparams( self, s, a, r, sp ):
		fs  = self.MDP.decode(  s, range( self.MDP.nstatefactors ) )
		fa  = self.MDP.decode(  a, range( self.MDP.nstatefactors, len( self.MDP.factordomains ) ) )
		fsp = self.MDP.decode( sp, range( self.MDP.nstatefactors ) )
		fsa = np.concatenate( ( fs, fa ) )

		compute_pi = False

		# update counts and sums for reward factors
		IR = self.MDP.rewardindices( fsa )
		for i in range( self.MDP.nrewardfactors ):
			self.NR[i][IR[i]] = self.NR[i][IR[i]] + 1
			self.SR[i][IR[i]] = self.SR[i][IR[i]] + r[i]

			self.v [i][IR[i]] = self.v [i][IR[i]] + 1
			if self.v[i][IR[i]] >= self.N_prev[i][IR[i]]:
				compute_pi = True

		# update counts for state factors
		IP = self.MDP.transitionindices( fsa )
		for i in range( self.MDP.nstatefactors ):
			self.NP[i][IP[i],fsp[i]] = self.NP[i][IP[i],fsp[i]] + 1

			self.v[self.MDP.nrewardfactors + i][IP[i]] = self.v[self.MDP.nrewardfactors + i][IP[i]] + 1
			if self.v[self.MDP.nrewardfactors + i][IP[i]] >= self.N_prev[self.MDP.nrewardfactors + i][IP[i]]:
				compute_pi = True

		return compute_pi

