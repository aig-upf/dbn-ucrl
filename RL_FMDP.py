
from __future__             import print_function

import sys
import time

from MDP                    import MDP
from FactoredMDP            import FactoredMDP

from UCRL                   import UCRL
from FactoredUCRL           import FactoredUCRL

from ConfidenceBounds       import *
from ExtendedValueIteration import ElementwiseEVI, OsbandEVI, StandardVI

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class RL_FMDP:

	def __init__( self, FMDP, time_horizon, ival, delta, no_expr ):
		self.FMDP         = FMDP
		self.time_horizon = time_horizon
		self.ival         = ival
		self.delta        = delta
		self.no_expr      = no_expr

	def run( self ):
		# Solve the full MDP
		MDP = self.FMDP.fullMDP()
		pi, g_opt = MDP.solve()

		print( g_opt, flush = True )

		# Main loop
		for k in range( self.no_expr ):
			print( k + 1, flush = True )

			UCRL_FactoredOsband  = FactoredUCRL( self.FMDP,
			                                     [OsbandRI()],
			                                     [OsbandCB()],
			                                     OsbandEVI() )

			PSRL_Factored        = FactoredUCRL( self.FMDP,
			                                     [PSRLTrueReward()],
			                                     [PSRLSampleKernel()],
			                                     StandardVI() )

			PSRL_Factored_G      = FactoredUCRL( self.FMDP,
			                                     [PSRLSampleReward()],
			                                     [PSRLSampleKernel()],
			                                     StandardVI() )

			DBN_UCRL             = FactoredUCRL( self.FMDP,
			                                     [HoeffdingLaplace()],
			                                     [BernoulliBernsteinPeeling(),EmpBernsteinPeeling()],
			                                     ElementwiseEVI() )

			UCRLB                =         UCRL( MDP,
			                                     [HoeffdingLaplace()],
			                                     [BernoulliBernsteinPeeling(),EmpBernsteinPeeling()],
			                                     ElementwiseEVI() )

			start = time.time()

			regret_UCRL_Factored_Osband = UCRL_FactoredOsband.runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of Factored Osband       : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of UCRL_Factored_Osband: " + str( regret_UCRL_Factored_Osband[-1] )  , flush = True )
			eprint( regret_UCRL_Factored_Osband                                                  , flush = True )

			start = time.time()

			regret_PSRL_Factored        = PSRL_Factored      .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of Factored PSRL         : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of PSRL_Factored       : " + str( regret_PSRL_Factored       [-1] )  , flush = True )
			eprint( regret_PSRL_Factored                                                         , flush = True )

			start = time.time()

			regret_PSRL_Factored_G      = PSRL_Factored_G    .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of Factored PSRL G       : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of PSRL_Factored_G     : " + str( regret_PSRL_Factored_G     [-1] )  , flush = True )
			eprint( regret_PSRL_Factored_G                                                       , flush = True )

			start = time.time()

			regret_DBN_UCRL             = DBN_UCRL           .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of DBN-UCRL              : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of DBN_UCRL            : " + str( regret_DBN_UCRL            [-1] )  , flush = True )
			eprint( regret_DBN_UCRL                                                              , flush = True )

			start = time.time()

			regret_UCRLB                = UCRLB              .runUCRL( self.delta, self.time_horizon, self.ival, g_opt )

			print ( "Time of UCRLB                 : " + str( time.time() - start ) + " seconds.", flush = True )
			print ( "Regret of UCRLB               : " + str( regret_UCRLB               [-1] )  , flush = True )
			eprint( regret_UCRLB                                                                 , flush = True )

