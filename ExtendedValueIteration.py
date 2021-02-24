
import numpy as np

class AbstractVI:
	def normalizeprobs( self, p_lower, p_hat, p_upper, V ):
		pass

	def computepolicy( self, M, r_upper, p_lower, p_hat, p_upper, t ):
		cnt    = 0
		V_old  = np.zeros ( M.nstates )
		V_diff = np.arange( M.nstates )

		while cnt <= 1000 and max( V_diff ) - min( V_diff ) > 1.0 / np.sqrt( t ):
			q      = self.normalizeprobs( p_lower, p_hat, p_upper, V_old )
			V, pi  = M.greedy( r_upper + q @ V_old )
			V_diff = V - V_old
			V_old  = V
			cnt    = cnt + 1

		return pi

# PSRL stores the sampled transition kernel in *p_upper*
class StandardVI( AbstractVI ):
	def normalizeprobs( self, p_lower, p_hat, p_upper, V ):
		return p_upper

class ElementwiseEVI( AbstractVI ):
	def normalizeprobs( self, p_lower, p_hat, p_upper, V ):
		# add confidence intervals to states in *decreasing* order of value
		tmp   = np.argsort( -V )
		q     = np.copy( p_lower )
		Delta = np.ones( np.size( q, 0 ) ) - np.sum( q, 1 )
		ix    = Delta > 1e-6
		s     = 0;

		while ix.any():
			new_Delta    = np.minimum( Delta[ix], p_upper[ix,tmp[s]] - q[ix,tmp[s]] );
			q[ix,tmp[s]] = q[ix,tmp[s]] + new_Delta
			Delta[ix]    = Delta[ix] - new_Delta
			ix           = Delta > 1e-6
			s            = s + 1

		return q

class OsbandEVI( AbstractVI ):
	def normalizeprobs( self, p_lower, p_hat, p_upper, V ):
		# add confidence intervals to states in *increasing* order of value
		tmp          = np.argsort( V )
		q            = np.copy( p_hat )
		q[:,tmp[-1]] = p_upper[:,tmp[-1]]
		ix           = np.sum( q, 1 ) > 1
		s            = 0;

		while ix.any():
			q[ix,tmp[s]] = np.maximum( 1 - np.sum( q[ix,:], 1 ) + q[ix,tmp[s]], 0 )
			ix           = np.sum( q, 1 ) > 1
			s 			 = s + 1;

		return q

