    # multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        foodList = newFood.asList()
        for food in foodList:
            dist = util.manhattanDistance(food, newPos)
            if dist != 0:
                score = score + (1.0/dist)

        ghostList = newGhostStates.asList()
        for ghost in ghostList:
            pos = ghost.getPosition()
            distance = util.manhattanDistance(pos, newPos)
            check = abs(newPos[0] - pos[0]) + abs(newPos[1] - pos[1])
            if check > 1:
                score = score + (1.0/distance)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def MiniMaxFunc(self, depth, pacmanIndex, gameState):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        store = []
        actions = gameState.getLegalActions(pacmanIndex)

        for act in actions:
            successor = gameState.generateSuccessor(pacmanIndex, act)
            if pacmanIndex + 1 >= gameState.getNumAgents():
                store += [self.MiniMaxFunc(depth + 1, 0, successor)]
            else:
                store += [self.MiniMaxFunc(depth, pacmanIndex + 1, successor)]

        if pacmanIndex == 0:
            if depth == 1:
                maximumScore = max(store)
                for i in range(len(store)):
                    if store[i] == maximumScore:
                        return actions[i]
            else:
                nodeValue = max(store)
        elif pacmanIndex > 0:
            nodeValue = min(store)
            
        return nodeValue

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.MiniMaxFunc(1, 0, gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def AlphaBetaFunc(self, depth, pacmanIndex, gameState, a, b):
        alpha = a
        beta = b

        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        store = []
        actions = gameState.getLegalActions(pacmanIndex)

        for act in actions:
            successor = gameState.generateSuccessor(pacmanIndex, act)

            if pacmanIndex + 1 >= gameState.getNumAgents():
                tmp = self.AlphaBetaFunc(depth + 1, 0, successor, alpha, beta)
            else:
                tmp = self.AlphaBetaFunc(depth, pacmanIndex + 1, successor, alpha, beta)

            if pacmanIndex == 0 and tmp > beta:
                return tmp
            if pacmanIndex > 0 and tmp < alpha:
                return tmp
            if pacmanIndex == 0 and tmp > alpha:
                alpha = tmp
            if pacmanIndex > 0 and tmp < beta:
                beta = tmp
            store += [tmp]

        if pacmanIndex == 0:
            if depth == 1:
                maximumScore = max(store)
                for i in range(len(store)):
                    if store[i] == maximumScore:
                        return actions[i]
            else:
                nodeValue = max(store)
        elif pacmanIndex > 0:
            nodeValue = min(store)

        return nodeValue

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBetaFunc(1, 0, gameState, -10000, 100000)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def ExpectimaxFunc(self, depth, pacmanIndex, gameState):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        store = []
        actions = gameState.getLegalActions(pacmanIndex)

        for act in actions:
            successor = gameState.generateSuccessor(pacmanIndex, act)
            if pacmanIndex + 1 >= gameState.getNumAgents():
                store += [self.ExpectimaxFunc(depth + 1, 0, successor)]
            else:
                store += [self.ExpectimaxFunc(depth, pacmanIndex + 1, successor)]

        if pacmanIndex == 0:
            if depth == 1:
                maximumScore = max(store)
                for i in range(len(store)):
                    if store[i] == maximumScore:
                        return actions[i]
            else:
                nodeValue = max(store)
        elif pacmanIndex > 0:
            x = sum(store)
            nodeValue = float(x / len(store))

        return nodeValue

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.ExpectimaxFunc(1, 0, gameState)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodPos = currentGameState.getFood().asList()
    foodDist = []
    curPos = list(currentGameState.getPacmanPosition())
    curScore = currentGameState.getScore()

    for food in foodPos:
        distance = util.manhattanDistance(food, curPos)
        foodDist.append(-1 * distance)
    if not foodDist:
        foodDist.append(0)

    return max(foodDist) + curScore


# Abbreviation
better = betterEvaluationFunction
