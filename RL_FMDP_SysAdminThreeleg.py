
from __future__             import print_function

import sys
import time

from UCRL                   import UCRL
from FactoredUCRL           import FactoredUCRL

from ConfidenceBounds       import *
from ExtendedValueIteration import ElementwiseEVI, OsbandEVI, StandardVI

from RiverSwimDomain        import RiverSwimDomain
from TwoRiverSwimDomain     import TwoRiverSwimDomain
from ThreeRiverSwimDomain   import ThreeRiverSwimDomain
from ModifiedCoffeeDomain   import ModifiedCoffeeDomain
from SysAdminCircularDomain import SysAdminCircularDomain
from SysAdminThreelegDomain import SysAdminThreelegDomain
from TaxiDomain             import TaxiDomain
from SmallTaxiDomain        import SmallTaxiDomain

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
	print( "Program is running...", flush = True )

	#FMDP = RiverSwimDomain( 3 )
	#FMDP = TwoRiverSwimDomain( 6 )
	#FMDP = ThreeRiverSwimDomain( 6 )
	#FMDP = ModifiedCoffeeDomain()
	FMDP = SysAdminCircularDomain( 7 )
	#FMDP = SysAdminThreelegDomain( 2 )
	#FMDP = TaxiDomain()
	#FMDP = SmallTaxiDomain()

	# Solve the full MDP
	MDP = FMDP.fullMDP()
	pi, g_opt = MDP.solve()

	print( g_opt, flush = True )

	# time horizon 10^5 in RiverSwim, SysAdmin, 10^6 - 10^7 in Coffee

	time_horizon = 100000
	ival         = 1
	delta        = 0.01
	no_expr      = 100

	# Main loop
	for k in range( no_expr ):
		print( k + 1, flush = True )

		UCRL_FactoredOsband  = FactoredUCRL( FMDP,
		                                     [OsbandRI()],
		                                     [OsbandCB()],
		                                     OsbandEVI() )

		PSRL_Factored        = FactoredUCRL( FMDP,
		                                     [PSRLTrueReward()],
		                                     [PSRLSampleKernel()],
		                                     StandardVI() )

		DBN_UCRL             = FactoredUCRL( FMDP,
		                                     [HoeffdingLaplace()],
		                                     [BernoulliBernsteinPeeling(),EmpBernsteinPeeling()],
		                                     ElementwiseEVI() )

		UCRLB                =         UCRL( MDP,
		                                     [HoeffdingLaplace()],
		                                     [BernoulliBernsteinPeeling(),EmpBernsteinPeeling()],
		                                     ElementwiseEVI() )

		start = time.time()

		regret_UCRL_Factored_Osband = UCRL_FactoredOsband .runUCRL( delta, time_horizon, ival, g_opt )

		print ( "Time of Factored Osband        : " + str( time.time() - start ) + " seconds.", flush = True )
		print ( "Regret of UCRL_Factored_Osband : " + str( regret_UCRL_Factored_Osband[-1] )  , flush = True )
		eprint( regret_UCRL_Factored_Osband                                                   , flush = True )

		start = time.time()

		regret_PSRL_Factored        = PSRL_Factored       .runUCRL( delta, time_horizon, ival, g_opt )

		print ( "Time of Factored PSRL          : " + str( time.time() - start ) + " seconds.", flush = True )
		print ( "Regret of PSRL_Factored        : " + str( regret_PSRL_Factored       [-1] )  , flush = True )
		eprint( regret_PSRL_Factored                                                          , flush = True )

		start = time.time()

		regret_DBN_UCRL             = DBN_UCRL            .runUCRL( delta, time_horizon, ival, g_opt )

		print ( "Time of DBN-UCRL               : " + str( time.time() - start ) + " seconds.", flush = True )
		print ( "Regret of DBN_UCRL             : " + str( regret_DBN_UCRL            [-1] )  , flush = True )
		eprint( regret_DBN_UCRL                                                               , flush = True )

		start = time.time()

		regret_UCRLB                = UCRLB               .runUCRL( delta, time_horizon, ival, g_opt )

		print ( "Time of UCRLB                  : " + str( time.time() - start ) + " seconds.", flush = True )
		print ( "Regret of UCRLB                : " + str( regret_UCRLB               [-1] )  , flush = True )
		eprint( regret_UCRLB                                                                  , flush = True )

	#print( "Elapsed time is " + str( time.time() - start ) + " seconds." )

if __name__ == "__main__":
    main()

