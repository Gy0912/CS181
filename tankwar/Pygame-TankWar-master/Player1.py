# player1.py

import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_size, action_size))
        self.log_file = open('training_log.txt', 'a')

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        if done and self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def log_training(self, level, score, lives):
        self.log_file.write(f"Level: {level}, Score: {score}, Lives: {lives}\n")
        self.log_file.flush()

def main(tank, position):
    state_size = 100  # Example state size; adjust as needed
    action_size = 5   # Example action size; adjust as needed
    agent = QLearningAgent(state_size, action_size)

    direction = tank.direction
    x, y = position

    state = get_state(x, y, direction)

    action = agent.choose_action(state)
    perform_action(tank, action)

def get_state(x, y, direction):
    # Implement your state representation logic here
    return np.random.randint(0, 100)

def perform_action(tank, action):
    if action == 0:
        tank.go_up()
    elif action == 1:
        tank.go_right()
    elif action == 2:
        tank.go_down()
    elif action == 3:
        tank.go_left()
    elif action == 4:
        tank.fire()
