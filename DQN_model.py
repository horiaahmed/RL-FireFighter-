import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from Fire_Fighter_Env import FireFighterEnv  

class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size, use_cnn=False):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.use_cnn = use_cnn
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.9  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.995  
        self.update_rate = 100  
        self.best_reward = -float("inf")
        self.best_weights = None 
        self.loss_history = []  

        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        model = Sequential()
        if self.use_cnn:
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.state_size[0], self.state_size[1], 1)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Flatten())
        else:
            model.add(Flatten(input_shape=self.state_size))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:  
            return random.randint(0, self.action_size-1)
        
        Q_values = self.main_network.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(Q_values[0])

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        minibatch = random.sample(self.replay_buffer, batch_size)
        total_loss = 0  

        for state, action, reward, next_state, done in minibatch:
            target_Q = reward + self.gamma * np.amax(self.target_network.predict(np.expand_dims(next_state, axis=0), verbose=0)) * (1 - done) 
            Q_values = self.main_network.predict(np.expand_dims(state, axis=0), verbose=0)
            Q_values[0][action] = target_Q

            history = self.main_network.fit(np.expand_dims(state, axis=0), Q_values, epochs=1, verbose=0)
            loss = history.history['loss'][0] 
            total_loss += loss 
        
        avg_loss = total_loss / batch_size 
        self.loss_history.append(avg_loss)  
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save_best_weights(self, total_reward):
       
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_weights = self.main_network.get_weights() 
            print(f"update best weights {total_reward}")

    def save_final_model(self):
        
        if self.best_weights is not None:
            self.main_network.set_weights(self.best_weights)
            self.main_network.save("Best_DQN_Model.keras")
            print("best mode is saved as Best_DQN_Model.keras'")

def plot_training(rewards_history, loss_history):
    plt.figure(figsize=(12, 5))

    # Plot Total Reward
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, label='Loss', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    env = FireFighterEnv()
state_size = (env.size[0], env.size[1])
action_size = 4 

num_episodes = 5
num_timesteps = 200
batch_size = 32
dqn = DQN(state_size, action_size, use_cnn=True)

rewards_history = []
time_step = 0

for episode in range(num_episodes):
    state = env.reset() 
    state = state['grid']
    total_reward = 0

    for t in range(num_timesteps):
        time_step += 1
        if time_step % dqn.update_rate == 0:
            dqn.update_target_network()

        action = dqn.epsilon_greedy(state)
        next_state, reward, done, _ = env.step(action)

        next_state = next_state['grid']  
        dqn.store_transition(state, action, reward, next_state, done)
        dqn.train(batch_size)

        state = next_state
        total_reward += reward

        if done:
            break

    dqn.save_best_weights(total_reward)
    rewards_history.append(total_reward)
    #print(f"Episode {episode}, Total Reward: {total_reward}, Avg Loss: {dqn.loss_history[-1] if dqn.loss_history else 0}")
    print("Episode: {}".format(episode), sep=' ')
    print("Total Reward: {}".format(total_reward) , sep=' ')
    print("Avg Loss: {}".format(dqn.loss_history[-1] if dqn.loss_history else 0) , sep=' ')

dqn.save_final_model()
plot_training(rewards_history, dqn.loss_history)
