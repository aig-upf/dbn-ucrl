
from RL_FMDP                import RL_FMDP

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
	FMDP = TwoRiverSwimDomain( 6 )
	#FMDP = ThreeRiverSwimDomain( 6 )
	#FMDP = ModifiedCoffeeDomain()
	#FMDP = SysAdminCircularDomain( 7 )
	#FMDP = SysAdminThreelegDomain( 2 )
	#FMDP = TaxiDomain()
	#FMDP = SmallTaxiDomain()

	time_horizon = 100000
	ival         = 10
	delta        = 0.01
	no_expr      = 100

	Experiment = RL_FMDP( FMDP, time_horizon, ival, delta, no_expr )
	Experiment.run()

if __name__ == "__main__":
    main()

