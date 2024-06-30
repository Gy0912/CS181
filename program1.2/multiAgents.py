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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodlist = newFood.asList()
        fooddistances = []
        ghostdistances = []
        inversefooddis = 0
        result = 1
        for (x,y) in foodlist:
            fooddistances += [abs(x - newPos[0]) + abs(y - newPos[1])]
            if min(fooddistances) != 0:
                inversefooddis = 1 / min(fooddistances)
        for ghost in newGhostStates:
            ghostposition = ghost.getPosition()
            ghostdistances += [abs(ghostposition[0] - newPos[0]) + abs(ghostposition[1] - newPos[1])]
        result = (inversefooddis) * (min(ghostdistances))**0.5
        currentghoststate = currentGameState.getGhostStates()
        currentscaredtime = [ghostState.scaredTimer for ghostState in currentghoststate]
        if childGameState.isLose():
            return -99999999
        if newScaredTimes[0] != 0:
            return 99999999
        if action == "Stop":
            result -= 10
        if currentscaredtime[0] != 0:
            result = inversefooddis - min(ghostdistances)
        result += (childGameState.getScore() - currentGameState.getScore())
        return result

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        
        def maxvalue(state, depth):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            value = -99999999
            for actions in state.getLegalActions(0):
                value = max(value, minvalue(state.getNextState(0, actions), depth, 1))
            return value
        
        def minvalue(state, depth, ghostnum):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            value = 99999999
            for actions in state.getLegalActions(ghostnum):
                if ghostnum == num - 1:
                    value = min(value, maxvalue(state.getNextState(ghostnum, actions), depth))
                else:
                    value = min(value, minvalue(state.getNextState(ghostnum, actions), depth, ghostnum + 1))
            return value
        
        maxtemp = -99999999
        finalaction = ''
        for actions in gameState.getLegalActions(0):
            tmp = minvalue(gameState.getNextState(0,actions), 0, 1)
            if tmp > maxtemp:
                maxtemp = tmp
                finalaction = actions
        return finalaction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        
        def maxvalue(state, depth, a, b):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            value = -99999999
            for actions in state.getLegalActions(0):
                value = max(value, minvalue(state.getNextState(0, actions), depth, 1, a, b))
                if value > b:
                    return value
                a = max(a, value)
            return value
        
        def minvalue(state, depth, ghostnum, a, b):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            value = 99999999
            for actions in state.getLegalActions(ghostnum):
                if ghostnum == num - 1:
                    value = min(value, maxvalue(state.getNextState(ghostnum, actions), depth, a, b))
                else:
                    value = min(value, minvalue(state.getNextState(ghostnum, actions), depth, ghostnum + 1, a, b))
                if value < a:
                    return value
                b = min(b, value)
            return value
        
        maxtemp = -99999999
        a = -99999999
        b = 99999999
        finalaction = ''
        for actions in gameState.getLegalActions(0):
            tmp = minvalue(gameState.getNextState(0,actions), 0, 1, a, b)
            if tmp > maxtemp:
                maxtemp = tmp
                finalaction = actions
            a = max(a, maxtemp)
        return finalaction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        
        def maxvalue(state, depth):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            value = -99999999
            for actions in state.getLegalActions(0):
                value = max(value, expvalue(state.getNextState(0, actions), depth, 1))
            return value
        
        def expvalue(state, depth, ghostnum):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            value = 0
            for actions in state.getLegalActions(ghostnum):
                if ghostnum == num - 1:
                    value += maxvalue(state.getNextState(ghostnum, actions), depth)
                else:
                    value += expvalue(state.getNextState(ghostnum, actions), depth, ghostnum + 1)
            return value
        
        maxtemp = -99999999
        finalaction = ''
        for actions in gameState.getLegalActions(0):
            tmp = expvalue(gameState.getNextState(0,actions), 0, 1)
            if tmp > maxtemp:
                maxtemp = tmp
                finalaction = actions
        return finalaction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    foodlist = newFood.asList()
    fooddistances = []
    ghostdistances = []
    result = currentGameState.getScore()
    for (x,y) in foodlist:
        fooddistances += [abs(x - newPos[0]) + abs(y - newPos[1])]
    if len(fooddistances) > 0:
        result += 1 / min(fooddistances)
    else:
        result += 1
    for ghost in newGhostStates:
        ghostposition = ghost.getPosition()
        ghostdistances = abs(ghostposition[0] - newPos[0]) + abs(ghostposition[1] - newPos[1])
        if ghostdistances > 0:
            if ghost.scaredTimer:
                result += 1000 / ghostdistances
            else:
                result -= 1 / ghostdistances
    return result

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState: GameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()