# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Initialize the qValues as an empty counter that will hold the Q-values for state-action pairs
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Retrieve the Q-value for the given state-action pair from qValues
        # Returns 0.0 if the pair has never been seen
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Get all legal actions for the current state
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0
        # Compute the maximum Q-value achievable from the current state
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state. Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Get all legal actions for the current state
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Find the best action with max Q-value
        bestValue = self.computeValueFromQValues(state)
        bestActions = [action for action in legalActions if self.getQValue(state, action) == bestValue]
        # Randomly choose between actions that have the best Q-value
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state. With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise. Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Get all legal actions for the current state
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Choose a random action based on probability of flipping coin
        # If True, it means it returned with probability of epsilon, then we choose a random legal action
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            # Otherwise, we stick with the best policy action
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Compute the current Q-value for the given state-action pair
        currentQValue = self.getQValue(state, action)
        # Compute the value of the next state
        nextValue = self.computeValueFromQValues(nextState)
        # Compute the sample based on received reward and future reward estimate
        rewardNextValueComb = reward + self.discount * nextValue
        # Update the Q-value for the state-action pair using the Q-learning formula
        self.qValues[(state, action)] = (1 - self.alpha) * currentQValue + self.alpha * rewardNextValueComb

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Retrieve the feature vector for the given state and action
        featureVector = self.featExtractor.getFeatures(state, action)
        # Calculate the QValue with weighted sum of feature functions and their weights
        QValue = 0
        for feature in featureVector:
            QValue += self.weights[feature] * featureVector[feature]
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Get the feature vector for the given state and action
        featureVector = self.featExtractor.getFeatures(state, action)
        # Calculate the current Q-value, which is Q(s, a)
        qValue = self.getQValue(state, action)
        # Compute the value of the next state
        nextValue = self.computeValueFromQValues(nextState)
        # Calculate difference term: difference = (r + discount * nextStateValue) - currentQValue
        difference = (reward + self.discount * nextValue) - qValue

        # Update the weights for each feature
        for feature in featureVector:
            self.weights[feature] += self.alpha * difference * featureVector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights)
