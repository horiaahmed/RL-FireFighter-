import numpy as np
import random
import tkinter as tk
from enum import Enum

class CellType(Enum):
    EMPTY = 0       # Empty land
    TREE = 1        # Healthy tree (🌳)
    FIRE = 2        # Burning tree (🔥)
    OBSTACLE = 3    # obstacle (🚧)

class FireFighterEnv:
    def __init__(self, size=9, fire_spawn_delay=10, max_steps=200):
        self.size = (size, size)            # Grid dimensions
        self.max_steps = max_steps          # Maximum allowed steps
        self.fire_spawn_delay = fire_spawn_delay  # Fire countdown timer
        
        # Initialize grid
        self.grid = np.zeros(self.size, dtype=np.int32)
        self.fire_timers = np.zeros(self.size, dtype=np.int32)
        self.agent_pos = (size//2, size//2)  # Start agent in center
        self.steps_left = max_steps
        self.score = 0
        
        # Generate forest and ensure agent mobility
        self._generate_forest()
        self._ignite_random_tree() 

    def _generate_forest(self):
        """Generate forest with 10% obstacles, 70% trees, 20% empty cells"""
        total_cells = self.size[0] * self.size[1]
        obstacles = int(total_cells * 0.1)    # 10% obstacles
        trees = int(total_cells * 0.7)        # 70% trees
        empty = total_cells - obstacles - trees  # 20% empty
        
        # Create and shuffle cells
        cells = (
            [CellType.OBSTACLE.value] * obstacles +
            [CellType.TREE.value] * trees +
            [CellType.EMPTY.value] * empty
        )
        random.shuffle(cells)
        
        # Fill grid
        index = 0
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.grid[x][y] = cells[index]
                index += 1
        
        # Ensure agent can move initially
        self._ensure_agent_mobility()
        
        # Ensure all trees have accessible paths
        self._ensure_tree_accessibility()

    def _ensure_tree_accessibility(self):
        """Ensure every tree has at least one adjacent non-obstacle cell as if this tree is fired then agent can extingish it."""
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.grid[x][y] == CellType.TREE.value:
                    # Check adjacent cells
                    has_non_obstacle = False
                    neighbors = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                            neighbors.append((nx, ny))
                            if self.grid[nx][ny] != CellType.OBSTACLE.value:
                                has_non_obstacle = True
                                break
                    if not has_non_obstacle and neighbors:
                        # Convert a random adjacent obstacle to empty
                        chosen = random.choice(neighbors)
                        self.grid[chosen[0]][chosen[1]] = CellType.EMPTY.value

    def _ensure_agent_mobility(self):
        """Ensure agent has at least one valid move in the start of playing by clearing obstacles """
        x, y = self.agent_pos
        
        # Possible movement directions
        directions = [
            (x-1, y),  # Up
            (x+1, y),  # Down
            (x, y-1),  # Left
            (x, y+1)   # Right
        ]
        
        # Check if agent is blocked
        blocked = True
        for nx, ny in directions:
            if (0 <= nx < self.size[0] and 
                0 <= ny < self.size[1] and 
                self.grid[nx][ny] != CellType.OBSTACLE.value):
                blocked = False
                break
        
        # If blocked, clear a random adjacent obstacle
        if blocked:
            valid_directions = [
                (nx, ny) for nx, ny in directions
                if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]
            ]
            if valid_directions:
                exit_x, exit_y = random.choice(valid_directions)
                self.grid[exit_x][exit_y] = CellType.EMPTY.value
                
    def _ignite_random_tree(self):
        """Start fire at random tree location (ignores obstacles)"""
        tree_positions = np.argwhere((self.grid == CellType.TREE.value) & 
                                    (self.fire_timers == 0))
        if len(tree_positions) > 0:
            x, y = random.choice(tree_positions)
            self.grid[x][y] = CellType.FIRE.value
            self.fire_timers[x][y] = self.fire_spawn_delay 
            
    def reset(self):
        """Reset environment to initial state"""
        self.__init__(size=self.size[0], 
                      fire_spawn_delay=self.fire_spawn_delay, 
                      max_steps=self.max_steps)
        return self._get_observation()
    
    def _get_observation(self):
        """Return current environment state"""
        return {
            'grid': self.grid.copy(),
            'agent_pos': self.agent_pos,
            'fire_timers': self.fire_timers.copy(),
            'steps_left': self.steps_left,
            'score': self.score
        }
    
    def step(self, action):
        """
        Execute one timestep of the environment
        Args:
            action: 0-3 for movement
        Returns:
            observation, reward, done, info
        """
        reward = 0
        done = False
        info = {}
        
        # Handle agent action
        movement_penalty =self._move_agent(action)
        reward += movement_penalty   # Time penalty for movement

      # Check if agent is on fire, extinguish it automatically
        x, y = self.agent_pos
        if self.grid[x][y] == CellType.FIRE.value:
            self._put_out_fire(x, y)
            reward += 100  # Reward for extinguishing fire
            
        # Update fire spread mechanics
        fire_spread_penalty = self._update_fire() 
        reward += fire_spread_penalty
        
        # Update counters
        self.steps_left -= 1
        self.score += reward
        
        # Check termination condition
        if self.steps_left <= 0:
            done = True
            info['reason'] = 'timeout'
            
        return self._get_observation(), reward, done, info
    
    def _move_agent(self, direction):
        """Move agent in specified direction with obstacle avoidance"""
        x, y = self.agent_pos
        new_pos = {
            0: (x-1, y),  # Up
            1: (x+1, y),  # Down
            2: (x, y-1),  # Left
            3: (x, y+1)   # Right
        }.get(direction, (x, y))
        
        # Validate new position
        if (0 <= new_pos[0] < self.size[0] and 
            0 <= new_pos[1] < self.size[1] and 
            self.grid[new_pos[0]][new_pos[1]] != CellType.OBSTACLE.value):

            self.agent_pos = new_pos
            
        return -1 
        
        # else:
        #     return -5
            
    def _extinguish_fire(self):
        """Extinguish fires in current and adjacent cells"""
        reward = 0
        x, y = self.agent_pos
        
        # Check current cell
        if self.grid[x][y] == CellType.FIRE.value:
            self._put_out_fire(x, y)
            reward += 100
            
        return reward
    
    def _put_out_fire(self, x, y):
        """Helper method for fire extinguishing"""
        self.grid[x][y] = CellType.EMPTY.value
        self.fire_timers[x][y] = 0
    
    def _update_fire(self):
        """Handle fire spread mechanics and return penalties"""
        penalty = 0
        new_fires = []
        
        # Process existing fires
        fire_cells = np.argwhere(self.grid == CellType.FIRE.value)
        for x, y in fire_cells:
            self.fire_timers[x][y] -= 1
            
            if self.fire_timers[x][y] <= 0:
                # Find flammable neighbors
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip current cell
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.size[0] and 
                            0 <= ny < self.size[1] and 
                            self.grid[nx][ny] == CellType.TREE.value):
                            neighbors.append((nx, ny))
                
                # Spread to random neighbor
                if neighbors:
                    nx, ny = random.choice(neighbors)
                    new_fires.append((nx, ny))
                    penalty -= 20  # Spread penalty
                                
                # Convert burned cell to empty
                self.grid[x][y] = CellType.EMPTY.value
                self.fire_timers[x][y] = 0
                penalty -= 50  # Destruction penalty
                
        # Ignite new fires
        for x, y in new_fires:
            self.grid[x][y] = CellType.FIRE.value
            self.fire_timers[x][y] = self.fire_spawn_delay
            
        # Ensure at least one active fire
        if len(np.argwhere(self.grid == CellType.FIRE.value)) == 0:
            self._ignite_random_tree()
            
        return penalty

class FireGameUI:
    """Tkinter-based GUI for visualizing the environment"""
    
    def __init__(self, root, env):
        self.env = env
        self.root = root
        self.root.title("Fire Forest")
        
        # Visualization parameters
        self.cell_size = 50
        self.agent_icon = "🚒"
        
        # Create main canvas
        self.canvas = tk.Canvas(root, 
                              width=self.env.size[1] * self.cell_size,
                              height=self.env.size[0] * self.cell_size,
                              bg="white")
        self.canvas.pack(padx=10, pady=10)
        
        # Information display
        self.info_label = tk.Label(root, 
                                 text=f"Score: 0 | Steps left: {env.max_steps}", 
                                 font=("Arial", 12))
        self.info_label.pack()
        
        self.render()

    def render(self,action=0):
        """Render current environment state"""
        self.canvas.delete("all")
        
        for x in range(self.env.size[0]):
            for y in range(self.env.size[1]):
                # Calculate cell coordinates
                x1 = y * self.cell_size
                y1 = x * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                cell = self.env.grid[x][y]
                bg_color = "white"  # Default obstacle color
                emoji = ""
                
                # Set cell appearance
                if cell == CellType.TREE.value:
                    bg_color = "#2d5a27"
                    emoji = "🌳"
                elif cell == CellType.FIRE.value:
                    bg_color = "#ff3300"
                    emoji = "🔥"
                elif cell == CellType.OBSTACLE.value:
                    bg_color = "#ffff99"
                    emoji = "🚧"
                else:
                    bg_color = "#404040"
                    emoji = ""
                
                # Draw cell background and content
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=bg_color, outline="gray")
                self.canvas.create_text((x1+x2)//2, (y1+y2)//2, 
                                      text=emoji, 
                                      font=("Arial", 16))
                
                # Draw agent
                if (x, y) == self.env.agent_pos:
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, 
                                         text=self.agent_icon, 
                                         font=("Arial", 16))
                
                # Display fire countdown
                if cell == CellType.FIRE.value:
                    timer = self.env.fire_timers[x][y]
                    self.canvas.create_text(x1 + 20, y1 + 20, 
                                           text=str(timer), 
                                           fill="white",
                                           font=("Arial", 10, "bold"))
                    
            dic = {
                0: 'Up',
                1: 'Down',
                2: 'Left',
                3: 'Right'
                }
            
        # Update score display
        self.info_label.config(text=f"Score: {self.env.score} | Steps left: {self.env.steps_left} | {dic[action]}")

    def update(self):
        """Update game state and visualization"""
        action = random.randint(0, 3) # Random policy for movement
        _, _, done, _ = self.env.step(action) # select random action of agent

        self.render(action)
        self.root.update_idletasks()
        self.root.update()

        if not done and self.env.steps_left > 0:
            self.root.after(1000, self.update) 

  
if __name__ == "__main__":
    # Initialize and run simulation
    env = FireFighterEnv(size=9, fire_spawn_delay=10, max_steps=200)
    root = tk.Tk()
    ui = FireGameUI(root, env)
    root.after(1000, ui.update)
    root.mainloop()