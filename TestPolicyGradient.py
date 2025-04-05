import numpy as np
import random
import tkinter as tk
import torch
from torch.distributions import Categorical
from Fire_Fighter_Env import FireFighterEnv, FireGameUI  # Assuming these are unchanged

# Load the PyTorch policy model
class PolicyNetwork(torch.nn.Module):
    def init(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).init()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

# Load the trained weights
model = PolicyNetwork(input_size=6, hidden_size=512, output_size=4)  # Match training architecture
model.load_state_dict(torch.load("best_policy.pt"))
model.eval()

# Function to convert state to tensor (from your PolicyGradientAgent)
def state_to_tensor(state_dict, size=9, max_steps=200):
    fire_locs = np.argwhere(state_dict['grid'] == 2)  # FIRE = 2 from CellType
    if len(fire_locs) > 0:
        agent_x, agent_y = state_dict['agent_pos']
        distances = np.sqrt(((fire_locs - [agent_x, agent_y])**2).sum(axis=1))
        nearest_idx = np.argmin(distances)
        nearest_fire_pos = fire_locs[nearest_idx]
    else:
        nearest_fire_pos = state_dict['agent_pos']
    
    agent_pos = torch.FloatTensor(state_dict['agent_pos']) / size
    fire_pos = torch.FloatTensor(nearest_fire_pos) / size
    steps_left = torch.FloatTensor([state_dict['steps_left'] / max_steps])
    score = torch.FloatTensor([state_dict['score'] / 1000.0])
    
    return torch.cat([agent_pos, fire_pos, steps_left, score]).unsqueeze(0)

# Initialize the environment
env = FireFighterEnv(size=9, fire_spawn_delay=10, max_steps=200)

# Initialize the UI
root = tk.Tk()
ui = FireGameUI(root, env)

# Initialize state and done flag
state = env.reset()
done = False
total_reward = 0

def run_episode():
    global state, done, total_reward
    if not done:
        # Sample an action from the policy
        state_tensor = state_to_tensor(state)
        with torch.no_grad():
            action_probs = model(state_tensor)  # Output is probabilities for each action
            m = Categorical(action_probs)
            action = m.sample().item()  # Sample action based on probabilities

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        state = next_state
        ui.render(action)

    # If episode ends, print total reward and reset
    if done:
        print(f"Episode completed with total reward: {total_reward}")
        state = env.reset()
        total_reward = 0
        done = False

    # Schedule the next step
    root.after(1000, run_episode)

# Start the first episode
root.after(0, run_episode)
root.mainloop()