# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Iterate for the specified number of iterations
        for _ in range(self.iterations):
            new_values = util.Counter()
            # Iterate over all states
            for state in self.mdp.getStates():
                # Check if the state is not a terminal state
                if not self.mdp.isTerminal(state):
                    # Get all possible actions from the current state
                    actions = self.mdp.getPossibleActions(state)
                    # Calculate the maximum Q-value among all actions and store in new values
                    new_values[state] = max(self.computeQValueFromValues(state, action) for action in actions)
            # Update the agent's value estimates with the new values
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0
        for nextState, transition_prob in self.mdp.getTransitionStatesAndProbs(state, action):
            """get the reward for going to the next state"""
            reward = self.mdp.getReward(state, action, nextState)
            """
              Compute Q value by multiplying transition probability with the sum of reward 
              for next state and discounted value of next state
            """
            q_value += transition_prob * (reward + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        """Get all the possible actions of current state"""
        actions = self.mdp.getPossibleActions(state)
        """if there are no legal actions, return None"""
        if not actions:
            return None
        """
          Compute the q_value for each action and return the best action
          based on the q_value
        """

        best_q_value = float('-inf')
        best_action = None

        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Use a dictionary of set for each state enables a O(1) search time complexity
        predecessors = {s: set() for s in self.mdp.getStates()}
        # Identify predecessors of each state
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, transitionProb in self.mdp.getTransitionStatesAndProbs(state, action):
                    # predecessors of a state s have non-zero transition probability
                    if transitionProb > 0:
                        predecessors[nextState].add(state)

        # Initialize empty priority queue
        priorityQueue = util.PriorityQueue()

        # Iterate over states in the order returned by self.mdp.getStates()
        for state in self.mdp.getStates():
            # only iterate for non-terminal state
            if not self.mdp.isTerminal(state):
                # Compute the max Qvalue across all possible actions from state (s)
                maxQValue = max(
                    self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                # Compute the absolute value of difference between current value of s and max Qvalue
                diff = abs(self.values[state] - maxQValue)
                priorityQueue.push(state, -diff)

        for _ in range(self.iterations):
            # if there's nothing left in the queue, we terminate
            if priorityQueue.isEmpty():
                break

            # Get the state s with the highest priority (highest error)
            state = priorityQueue.pop()
            if not self.mdp.isTerminal(state):
                # Update the value of the state (s)
                maxQValue = max(
                    self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                self.values[state] = maxQValue

            # Update the priorities of predecessor states (p)
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    # Compute the max Qvalue across all possible actions from p
                    maxQValue = max(
                        self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p))
                    diff = abs(self.values[p] - maxQValue)
                    if diff > self.theta:
                        # if not in the queue, simply act as push
                        # if it already exists in the queue, we update the diff to make sure
                        # there is no such p in the queue that has lower or equal priority (diff)
                        priorityQueue.update(p, -diff)
