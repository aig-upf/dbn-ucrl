
import numpy as np
from AbstractMDP import AbstractMDP
from MDP import MDP

class FactoredMDP( AbstractMDP ):

	def __init__( self, nstatefactors, nrewardfactors, factordomains ):
		self.nstatefactors  = nstatefactors
		self.nrewardfactors = nrewardfactors
		self.factordomains  = np.array( factordomains )

		super().__init__( np.prod( self.statedomains() ), np.prod( self.actiondomains() ) )

		self.statemappings    = [None] * nstatefactors
		self.rewardstruct     = [None] * nrewardfactors
		self.transitionstruct = [None] * nstatefactors

		for i in range( nstatefactors ):
			self.statemappings[i] = np.zeros( ( factordomains[i], self.nelements( range( nstatefactors ) ) ) )

		for s in range( self.nelements( range( nstatefactors ) ) ):
			state = self.decode( s, range( nstatefactors ) )
			for i in range( nstatefactors ):
				self.statemappings[i][state[i]][s] = 1

	def assignmappings( self ):
		for sa in range( self.nelements( range( len( self.factordomains ) ) ) ):
			stateAction = self.decode( sa, range( len( self.factordomains ) ) )
			for i in range( self.nrewardfactors ):
				self.rewardstruct[i].mapping[sa, self.encode( stateAction, self.rewardstruct[i].scope )] = 1
			for i in range( self.nstatefactors ):
				self.transitionstruct[i].mapping[sa, self.encode( stateAction, self.transitionstruct[i].scope )] = 1

	def nelements( self, scope ):
		return np.prod( [self.factordomains[key] for key in scope] )

	def encode( self, state, scope ):
		return np.ravel_multi_index( state[scope], self.factordomains[scope] )

	def decode( self, s, scope ):
		return np.array( np.unravel_index( s, self.factordomains[scope] ) )

	def statedomains( self ):
		return self.factordomains[range( self.nstatefactors )]

	def actiondomains( self ):
		return self.factordomains[range( self.nstatefactors, len( self.factordomains ) )]

	def rewardsizes( self ):
		return [self.nelements( self.rewardstruct[i].scope ) for i in range( self.nrewardfactors )]

	def transitionsizes( self ):
		return [self.nelements( self.transitionstruct[i].scope ) for i in range( self.nstatefactors )]

	def rewardindices( self, stateAction ):
		return [self.encode( stateAction, self.rewardstruct[i].scope ) for i in range( self.nrewardfactors )]

	def transitionindices( self, stateAction ):
		return [self.encode( stateAction, self.transitionstruct[i].scope ) for i in range( self.nstatefactors )]

	def act( self, state, action ):
		fstate             = self.decode( state , range( self.nstatefactors ) )
		faction            = self.decode( action, range( self.nstatefactors, len( self.factordomains ) ) )
		fnextstate, reward = self.factoredact( np.concatenate( ( fstate, faction ) ) )
		return self.encode( fnextstate, range( self.nstatefactors ) ), reward

	def factoredact( self, fsa ):
		nextstate = [0] * self.nstatefactors
		for i in range( self.nstatefactors ):
			sa = self.encode( fsa, self.transitionstruct[i].scope )
			nextstate[i] = np.random.choice( np.arange( self.factordomains[i] ), p = self.transitionstruct[i].params[sa,:] )

		reward = [0] * self.nrewardfactors
		for i in range( self.nrewardfactors ):
			sa = self.encode( fsa, self.rewardstruct[i].scope )
			reward[i] = self.rewardstruct[i].params[sa]

		return np.array( nextstate ), np.array( reward )

	def fullMDP( self ):
		rewards = np.zeros( self.nstates * self.nactions )
		for i in range( self.nrewardfactors ):
			rewards = rewards + self.rewardstruct[i].mapping @ self.rewardstruct[i].params

		kernel = np.ones( ( self.nstates * self.nactions, self.nstates ) )
		for i in range( self.nstatefactors ):
			kernel = kernel * ( self.transitionstruct[i].mapping @ self.transitionstruct[i].params @ self.statemappings[i] )

		return MDP( self.nstates, self.nactions, rewards, kernel )

	def resetstate( self ):
		pass

