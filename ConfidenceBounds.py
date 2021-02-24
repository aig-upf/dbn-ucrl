
import numpy as np
import scipy.stats as stat

# In these classes we do not worry about the magnitude of the confidence bounds!
# The method transitionbound in AbstractUCRL ensures that transition probabilities are in [0,1]

class AbstractCB:
	def ell( self, Nplus, delta ):
		# Nplus should have minimum values 2!!!
		eta   = 1.12
		return eta * np.log( np.log( Nplus ) * np.log( eta * Nplus ) / ( delta * np.log( eta ) ** 2 ) )

	def beta( self, Nplus, delta ):
		return np.sqrt( 2 * ( 1.0 + 1.0 / Nplus ) * np.log( np.sqrt( Nplus + 1 ) / delta ) / Nplus )

	def search_down( self, f, up, down, epsilon = 0.0001 ):
		if ( up - down <= epsilon ).all():
			return np.where( f( down ), down, up )
		mid  = ( up + down ) / 2
		fmid = f( mid )
		return self.search_down( f, np.where( fmid, mid, up ), np.where( fmid, down, mid ) )

	def search_up( self, f, up, down, epsilon = 0.0001 ):
		if ( up - down <= epsilon ).all():
			return np.where( f( up ), up, down )
		mid = ( up + down ) / 2
		fmid = f( mid )
		return self.search_up( f, np.where( fmid, up, mid ), np.where( fmid, mid, down ) )

	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		pass

class EmpBernsteinPeeling( AbstractCB ):
	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		Nplus = np.maximum( Nplus, 2 )
		ell = self.ell( Nplus, delta ) / Nplus

		# if v_est is a matrix, we have to reshape ell
		ell = ell[:,None] if len( v_est.shape ) > 1 else ell

		d = np.sqrt( 2 * v_est * ( 1 - v_est ) * ell ) + 7 * ell / 3
		return v_est - d, v_est + d

class BernoulliBernsteinPeeling( AbstractCB ):
	def localbound( self, q, ell ):
		return np.sqrt( 2 * q * ( 1 - q ) * ell ) + ell / 3

	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		Nplus = np.maximum( Nplus, 2 )
		ell = self.ell( Nplus, delta ) / Nplus

		# if v_est is a matrix, we have to reshape ell
		ell = ell[:,None] if len( v_est.shape ) > 1 else ell

		LB = self.search_down( lambda q: v_est - q <= self.localbound( q, ell ), v_est, np.zeros( v_est.shape ) )
		UB = self.search_up  ( lambda q: q - v_est <= self.localbound( q, ell ), np.ones ( v_est.shape ), v_est )

		return LB, UB

class OsbandRI( AbstractCB ):
	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		C_r   = np.log( 4 * t / delta )
		d     = np.sqrt( C_r / Nplus )

		# if v_est is a matrix, we have to reshape d
		d = d[:,None] if len( v_est.shape ) > 1 else d

		return v_est - d, v_est + d

class OsbandCB( AbstractCB ):
	# what should S be if v_est is reward?
	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		C_p   = 4 * S * np.log( 4 * t / delta )
		d     = np.sqrt( C_p / Nplus )

		# if v_est is a matrix, we have to reshape d
		d = d[:,None] if len( v_est.shape ) > 1 else d

		return v_est - d, v_est + d

class L1Laplace( AbstractCB ):
	# what should S be if v_est is reward? What if 2 ** S is too large?
	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		C_p   = 2 * np.log( ( 2 ** S - 2 ) * 2 / delta )
		d     = np.sqrt( ( 1.0 + 1.0 / Nplus ) * ( C_p + np.log( Nplus + 1 ) ) / Nplus )

		# if v_est is a matrix, we have to reshape d
		d = d[:,None] if len( v_est.shape ) > 1 else d

		return v_est - d, v_est + d

class HoeffdingLaplace( AbstractCB ):
	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		beta = self.beta( Nplus, delta ) / 2

		# if v_est is a matrix, we have to reshape beta
		beta = beta[:,None] if len( v_est.shape ) > 1 else beta

		return v_est - beta, v_est + beta

class HoeffdingSubGaussianLaplace( AbstractCB ):
	# in our case, variance of reward is 0! (we do not add noise to reward)
	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		Nplus = np.maximum( Nplus, 2 )
		ell = self.ell( Nplus, delta )
		d   = 7 * ell / ( 3 * Nplus )

		# if v_est is a matrix, we have to reshape d
		d = d[:,None] if len( v_est.shape ) > 1 else d

		return v_est, v_est

class BernoulliSubGaussianLaplace( AbstractCB ):
	def g( self, p ):
		valid         = np.logical_or( np.logical_and( 0 < p, p < 0.5 ), np.logical_and( 0.5 < p, p < 1 ) )
		res           = np.zeros( p.shape )
		res[p == 0.5] = 0.25
		res[valid]    = ( 0.5 - p[valid] ) / np.log( 1 / p[valid] - 1 )
		return res

	def gbar( self, p ):
		return np.where( p < 0.5, self.g( p ), p * ( 1 - p ) )

	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		beta = self.beta( Nplus, delta )

		# if v_est is a matrix, we have to reshape beta
		beta = beta[:,None] if len( v_est.shape ) > 1 else beta

		LB = self.search_down( lambda q: v_est-q <= beta*np.sqrt(self.g   (q)), v_est, np.zeros( v_est.shape ) )
		UB = self.search_up  ( lambda q: q-v_est <= beta*np.sqrt(self.gbar(q)), np.ones ( v_est.shape ), v_est )

		return LB, UB

class PSRLSampleReward( AbstractCB ):

	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		r_sampled = stat.norm.rvs( v_est, 1 / Nplus )
		return v_est, r_sampled

class PSRLTrueReward( AbstractCB ):

	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		return v_est, N

class PSRLSampleKernel( AbstractCB ):

	def confidencebound( self, S, v_est, Nplus, N, delta, t ):
		p_sampled = np.array( [stat.dirichlet.rvs( row + 1 ).flatten() for row in N] )
		return v_est, p_sampled

