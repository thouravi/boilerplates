import numpy as np
import random
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam

# Define the reinforcement learning agent
class Agent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = []
    self.gamma = 0.95  # Discount rate
    self.epsilon = 1.0  # Exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.model = self._build_model()

  def _build_model(self):
    # Build a neural network to approximate the Q-function
    model = Sequential()
    model.add(Dense(32, input_dim=self.state_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return model

  def remember(self, state, action, reward, next_state, done):
    # Store a transition in the memory
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    # Choose an action in the current state based on the current policy
    if np.random.rand() <= self.epsilon:
      return np.random.randint(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # Returns the index of the maximum value in the array

  def replay(self, batch_size):
    # Sample a minibatch of transitions from the memory
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
      target_f = self.model.predict(state)
      target_f[0][action] = target
      self.model.fit(state, target_f, epochs=1, verbose=0)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
  
  def reward(self, state, action, reward, next_state, done):
    # Compute the reward for a given transition
    target = reward
    if not done:
      target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
    return target

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)
