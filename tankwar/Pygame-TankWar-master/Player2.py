# from Enhance import PlayEnhance
# from CommonHeader import *
# """
#     TILE_BRICK:    砖块
#     TILE_STEEL:    铁
#     TILE_WATER:    水
#     ILE_GRASS:     草
#     TILE_FROZE:    冰
#     TANK_PLAYER1:  玩家1
#     TANK_PLAYER2:  玩家2
#     TANK_PLAYER3:  敌人
#     TANK_BULLET :  子弹
    
#     DIR_UP
#     DIR_RIGHT
#     DIR_DOWN
#     DIR_LEFT
# """
# TANK_CURRENT = 51


# def main(tank, position):
#     """
#     玩家1 在此游戏
#     """

#     # demo
#     # get my direciton
#     direction = tank.direction
#     # get my position
#     x, y = position

#     # search enemy towards
#     enemies = PlayEnhance.find_enemy_towards(tank, direction)
#     if len(enemies) != 0:
#         tank.fire()

#     for enemy in PlayEnhance.get_all_enemies():
#         print(PlayEnhance.get_position(enemy))

#     # search buttlets from north
#     bullets = PlayEnhance.find_element(tank, DIR_UP, TANK_BULLET)
#     for bullet in bullets:
#         if bullet[2].direction == DIR_DOWN:
#             tank.go_left()

# import numpy as np
# import random
# from collections import deque
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from Enhance import PlayEnhance
# from CommonHeader import *

# # DQN Agent
# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()

#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#                       optimizer=Adam(learning_rate=self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = (reward + self.gamma *
#                           np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# # Constants
# STATE_SIZE = 8  # Example state size, you need to define it according to your game
# ACTION_SIZE = 4  # Up, Down, Left, Right

# # Initialize agent
# agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# def get_state(tank):
#     # Example state: [tank x, tank y, tank direction, enemy positions, bullet positions]
#     state = [tank.rect.left, tank.rect.top, tank.direction]
#     for enemy in PlayEnhance.get_all_enemies():
#         state.extend(PlayEnhance.get_position(enemy))
#     for bullet in PlayEnhance.bullets:
#         state.extend(PlayEnhance.get_position(bullet))
#     return np.reshape(state, [1, STATE_SIZE])

# def main(tank, position):
#     """
#     玩家1 在此游戏
#     """
#     state = get_state(tank)
#     action = agent.act(state)
#     next_state, reward, done = None, None, None  # You need to define how to get these values

#     # Perform the chosen action
#     if action == 0:
#         tank.go_up()
#     elif action == 1:
#         tank.go_right()
#     elif action == 2:
#         tank.go_down()
#     elif action == 3:
#         tank.go_left()

#     next_state = get_state(tank)
#     # Define the reward function according to your game logic
#     reward = 0
#     if len(PlayEnhance.find_enemy_towards(tank, tank.direction)) > 0:
#         tank.fire()
#         reward = 10
#     if tank.state == STATE_DEAD:
#         done = True
#         reward = -100

#     agent.remember(state, action, reward, next_state, done)
#     agent.replay(32)  # Train the agent with a minibatch size of 32


import numpy as np
import random
import pickle
from Enhance import PlayEnhance
from CommonHeader import *
import pygame
# from BattleCity import Game, Castle, Bullet, Tank, players, enemies, sounds, bullets, bonuses, labels, tank_position
# from FrameState import OnPlaying
# from Interval import gtimer

# Q-learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_size, action_size))

    def get_state(self, tank):
        state = [tank.rect.left // 16, tank.rect.top // 16, tank.direction]
        for enemy in PlayEnhance.get_all_enemies():
            state.extend(PlayEnhance.get_position(enemy))
        for bullet in PlayEnhance.bullets:
            state.extend(PlayEnhance.get_position(bullet))
        return hash(tuple(state)) % self.state_size

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

# Constants
STATE_SIZE = 10000  # Example state size, you need to define it according to your game
ACTION_SIZE = 4  # Up, Down, Left, Right

# Initialize agent
agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)

def get_state(tank):
    state = [tank.rect.left // 16, tank.rect.top // 16, tank.direction]
    for enemy in PlayEnhance.get_all_enemies():
        state.extend(PlayEnhance.get_position(enemy))
    for bullet in PlayEnhance.bullets:
        state.extend(PlayEnhance.get_position(bullet))
    return hash(tuple(state)) % STATE_SIZE

def check_collision(tank, direction):
    next_position = list(tank.rect.topleft)
    if direction == DIR_UP:
        next_position[1] -= tank.speed
    elif direction == DIR_RIGHT:
        next_position[0] += tank.speed
    elif direction == DIR_DOWN:
        next_position[1] += tank.speed
    elif direction == DIR_LEFT:
        next_position[0] -= tank.speed
    next_rect = pygame.Rect(next_position, tank.rect.size)
    for obstacle in PlayEnhance.game.level.obstacle_rects:
        if next_rect.colliderect(obstacle):
            return True
    return False

def main(tank, position):
    state = get_state(tank)
    action = agent.choose_action(state)

    if action == 0 and not check_collision(tank, DIR_UP):
        tank.go_up()
    elif action == 1 and not check_collision(tank, DIR_RIGHT):
        tank.go_right()
    elif action == 2 and not check_collision(tank, DIR_DOWN):
        tank.go_down()
    elif action == 3 and not check_collision(tank, DIR_LEFT):
        tank.go_left()

    next_state = get_state(tank)
    reward = 0
    done = False

    if len(PlayEnhance.find_enemy_towards(tank, tank.direction)) > 0:
        tank.fire()
        reward = 10
    if tank.state == STATE_DEAD:
        done = True
        reward = -100

    if not done:
        if action == 0 and not check_collision(tank, DIR_UP):
            reward += 1
        elif action == 1 and not check_collision(tank, DIR_RIGHT):
            reward += 1
        elif action == 2 and not check_collision(tank, DIR_DOWN):
            reward += 1
        elif action == 3 and not check_collision(tank, DIR_LEFT):
            reward += 1
        else:
            reward -= 1

    agent.learn(state, action, reward, next_state)

# Train the agent
def train_agent(agent, episodes=1000):
    for e in range(episodes):
        game = Game()
        castle = Castle()
        game.showMenu()

        game_running = True
        while game_running:
            time_passed = game.clock.tick(50)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pass
                elif event.type == pygame.QUIT:
                    game_running = False
                    quit()
                elif event.type == pygame.KEYDOWN and not game.game_over and game.active:
                    if event.key == pygame.K_m:
                        play_sounds = not play_sounds
                        if not play_sounds:
                            pygame.mixer.stop()
                        else:
                            sounds["bg"].play(-1)

                    for player in players:
                        if player.state == player.STATE_ALIVE:
                            try:
                                index = player.controls.index(event.key)
                            except:
                                pass
                            else:
                                if index == 0:
                                    if player.fire() and play_sounds:
                                        sounds["fire"].play()
                                elif index == 1:
                                    player.pressed[0] = True
                                elif index == 2:
                                    player.pressed[1] = True
                                elif index == 3:
                                    player.pressed[2] = True
                                elif index == 4:
                                    player.pressed[3] = True
                elif event.type == pygame.KEYUP and not game.game_over and game.active:
                    for player in players:
                        if player.state == player.STATE_ALIVE:
                            try:
                                index = player.controls.index(event.key)
                            except:
                                pass
                            else:
                                if index == 1:
                                    player.pressed[0] = False
                                elif index == 2:
                                    player.pressed[1] = False
                                elif index == 3:
                                    player.pressed[2] = False
                                elif index == 4:
                                    player.pressed[3] = False

            PlayEnhance.enemies = enemies
            PlayEnhance.bullets = bullets
            OnPlaying.game_running(players, enemies, bullets, bonuses)

            for player in players:
                if player.state == player.STATE_ALIVE and not game.game_over and game.active:
                    main(player, tank_position(player))

                    if player.pressed[0] == True:
                        player.move(DIR_UP)
                    elif player.pressed[1] == True:
                        player.move(DIR_RIGHT)
                    elif player.pressed[2] == True:
                        player.move(DIR_DOWN)
                    elif player.pressed[3] == True:
                        player.move(DIR_LEFT)
                player.update(time_passed)

            for enemy in enemies:
                if enemy.state == STATE_DEAD and not game.game_over and game.active:
                    enemies.remove(enemy)
                    if len(game.level.enemies_left) == 0 and len(enemies) == 0:
                        game.finishLevel()
                else:
                    enemy.update(time_passed)

            if not game.game_over and game.active:
                for player in players:
                    if player.state == STATE_ALIVE:
                        if player.bonus != None and player.side == Tank.SIDE_PLAYER:
                            game.triggerBonus(bonus, player)
                            player.bonus = None
                    elif player.state == STATE_DEAD:
                        player.superpowers = 0
                        player.lives -= 1
                        if player.lives > 0:
                            game.respawnPlayer(player)
                        else:
                            game.gameOver()

            for bullet in bullets:
                if bullet.state == Bullet.STATE_REMOVED:
                    bullets.remove(bullet)
                else:
                    bullet.update()

            for bonus in bonuses:
                if not bonus.active:
                    bonuses.remove(bonus)

            for label in labels:
                if not label.active:
                    labels.remove(label)

            if not game.game_over:
                if not castle.active:
                    game.gameOver()

            gtimer.update(time_passed)
            game.draw()

        agent.save('q_table.pkl')

if __name__ == "__main__":
    agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)
    try:
        agent.load('q_table.pkl')
    except FileNotFoundError:
        pass
    train_agent(agent)
