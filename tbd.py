import rasterio
import numpy as np


# Path to the DEM file
# dem_path = 'D:\\adity\\Downloads\\b01_009894_1665_xi_13s042w_b02_010606_1666_xn_13s042w_tied-dem.tif'
# https://asc-pds-individual-investigations.s3.us-west-2.amazonaws.com/mars_mro_hirise_explorationzones_day_2023/index.html
dem_path = 'b18_016833_1714_xi_08s124w_g02_019035_1714_xn_08s124w_tied-dem.tif'



def replace_outliers_with_adjacent_average(arr, threshold):
    non_zero_values = arr[arr != 0]
    
    # Calculate mean and standard deviation of non-zero values
    mean_non_zero = np.mean(non_zero_values)
    std_non_zero = np.std(non_zero_values)
    
    # Identify values within one standard deviation of the mean
    within_std_values = non_zero_values[np.abs(non_zero_values - mean_non_zero) <= std_non_zero]
    
    # Calculate the mean of these values
    mean_within_std = np.mean(within_std_values)

    outliers = np.abs(arr - mean_within_std) > threshold * np.std(arr)  # Identify outliers using z-score
    
    
    # Iterate over each outlier
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if outliers[i, j]:
                adjacent_values = []
                # Collect adjacent values (horizontally and vertically)
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= i + di < arr.shape[0] and 0 <= j + dj < arr.shape[1]:
                        adjacent_values.append(arr[i + di, j + dj])
                # Replace outlier with the average of adjacent values
                # arr[i, j] = np.mean(arr[arr != 0])
                arr[i, j] = mean_within_std
    
    return arr

# Specify threshold for z-score (e.g., 2 for removing values > 2 standard deviations from mean)
threshold = 2

with rasterio.open(dem_path) as dataset:
    dem_data = dataset.read(1)  # Assuming the DEM data is in the first band
    transform = dataset.transform

# Print the shape of the DEM data
print("DEM data shape:", dem_data.shape)



height, width = dem_data.shape
x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
# print(x_indices)
print(np.shape(x_indices))
# print(y_indices)

# # Apply the affine transformation to the grid indices
x_coords = transform[0] * x_indices + transform[1] * y_indices + transform[2]
y_coords = transform[3] * x_indices + transform[4] * y_indices + transform[5]


# # Print the first few values to verify
# print("Sample x_coords:\n", x_coords[:5, :5])
# print("Sample y_coords:\n", y_coords[:5, :5])


# # Normalize the coordinates
x_coords -= x_coords.min()
y_coords -= y_coords.min()

# # Scale the coordinates (if needed)
x_coords /= 18
y_coords /= 18




# (1445, 1905)

# row_start, row_end = 395, 745
# col_start, col_end = 700, 1050

# row_start, row_end = 700, 900
# col_start, col_end = 700, 900

# row_start, row_end = 600, 900
# col_start, col_end = 600, 900

row_start, row_end = 1200, 1355
col_start, col_end = 250, 600



# # Slice the data
dem_data_subset = dem_data[row_start:row_end, col_start:col_end]
x_coords_subset = x_coords[row_start:row_end, col_start:col_end]
y_coords_subset = y_coords[row_start:row_end, col_start:col_end]

# dem_data_subset_normalized = (dem_data_subset - dem_data_subset.min()) / (dem_data_subset.max() - dem_data_subset.min())


# Replace outliers with the average of adjacent values
dem_data_subset_cleaned= replace_outliers_with_adjacent_average(dem_data_subset, threshold)

print(dem_data_subset_cleaned.max())
print(dem_data_subset_cleaned.min())









import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback,  StopTrainingOnNoModelImprovement, BaseCallback
import tensorflow as tf


class MarsRoverEnv(gym.Env):
    def __init__(self, grid_size = dem_data_subset_cleaned.shape ,start = (10, 10), goal = (120, 339), kd=1.0, kh=30.0, kr=15.0):
        super(MarsRoverEnv, self).__init__()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.kd = kd
        self.kh = kh
        self.kr = kr
        self.terrain = dem_data_subset_cleaned
        self.state = start
        self.visited_cells = set()  
        self.previous_distance = np.linalg.norm(np.array(start) - np.array(goal))
        self.step_penalty = -1 
        
        self.action_space = spaces.Discrete(8)  # 8 possible movements (up, down, left, right, and diagonals)
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.visited_cells = set()
        self.previous_distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
        return np.array(self.state, dtype=np.int64), {}

    def step(self, action):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        next_state = (self.state[0] + actions[action][0], self.state[1] + actions[action][1])
        
        if 0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]:
            energy_cost = self.calculate_energy_cost(self.state[0], self.state[1], next_state[0], next_state[1])
            current_distance = np.linalg.norm(np.array(next_state) - np.array(self.goal))
            distance_reward = 25.0 * (self.previous_distance - current_distance)
            # print( f"Energy Cost: {energy_cost}")
            # print(f"Distance reward: {distance_reward}")
            reward = -energy_cost + distance_reward + self.step_penalty
            if self.state in self.visited_cells:
                reward = -15 

            else:
                self.visited_cells.add(self.state)

            self.previous_distance = current_distance
            self.state = next_state

            if next_state == self.goal:
                reward += 200  # Large reward for reaching the goal
                terminated = True
            else:
                terminated = False
        else:
            reward = -30  # Penalty for invalid moves
            terminated = False

        truncated = False  # No truncation in this case
        info = {}

        return np.array(self.state, dtype=np.int64), reward, terminated, truncated, info

    def calculate_energy_cost(self, x1, y1, x2, y2):
        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        h = self.terrain[x2, y2] - self.terrain[x1, y1]
        if h > 0:
            return self.kd * d + self.kh * h  # Uphill movement cost
        else:
            return self.kd * d - self.kr * abs(h)  # Downhill movement cost (with energy regeneration)

    def render(self, mode='human'):
        plt.imshow(self.terrain, cmap='terrain', interpolation='bilinear')
        plt.colorbar(label='Elevation')
        plt.scatter(self.start[1], self.start[0], color='red', marker='o', label='Start')
        plt.scatter(self.goal[1], self.goal[0], color='green', marker='x', label='Goal')
        plt.scatter(self.state[1], self.state[0], color='blue', marker='o', label='Rover')
        plt.legend()
        plt.show()


import os

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, eval_freq: int, log_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.save_path = os.path.join(log_path, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0 and 'r' in ep_info_buffer[0]:
                mean_reward = np.mean([ep_info['r'] for ep_info in ep_info_buffer])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model_test'))
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path} with mean reward {self.best_mean_reward}")
        return True

# Initialize the environment

env = MarsRoverEnv()
# env = Monitor(env)

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="ppo_logs")

eval_callback = SaveOnBestTrainingRewardCallback( eval_freq=2000, log_path="logs2")

# eval_callback = EvalCallback(env, best_model_save_path="./logs2/",
#                              log_path="./logs2/", eval_freq=3000,
#                              deterministic=True, render=False)


# Train the model
# Monitor performance metrics
timesteps = 5000
results = model.learn(total_timesteps=timesteps, callback=eval_callback)

# Save the model
model.save(f"ppo_mars_rover_{timesteps}_test")

# Load the model
model = PPO.load(f"ppo_mars_rover_{timesteps}_test")

# Evaluate the trained model
obs, _ = env.reset()
ppo_energy_cost = 0
ppo_path = [env.start]
count_steps=0
for _ in range(100000):
    action, _states = model.predict(obs)
    next_obs, rewards, terminated, truncated, info = env.step(action)
    # if(rewards>0):
    #     print(rewards)
    ppo_energy_cost += env.calculate_energy_cost(obs[0], obs[1], next_obs[0], next_obs[1])
    ppo_path.append((next_obs[0], next_obs[1]))
    obs = next_obs
    if terminated or truncated:
        break
    count_steps+=1

# Print energy costs
print(f"Refined Path Energy Cost (PPO): {ppo_energy_cost}")
print(f"PPo steps: {count_steps}")




import heapq

def dijkstra_path(terrain, start, goal):
    grid_size = terrain.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_list:
        current_cost, current = heapq.heappop(open_list)
        if current == goal:
            break
        for action in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + action[0], current[1] + action[1])
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]:
                tentative_g_score = g_score[current] + env.calculate_energy_cost(current[0], current[1], neighbor[0], neighbor[1])
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_list, (tentative_g_score, neighbor))
    
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    total_energy_cost = g_score[goal]
    return path, total_energy_cost

# Define the initial A* path and energy cost calculation
def a_star_path(start, goal, terrain, kd, kh, kr):
    from queue import PriorityQueue
    grid_size = terrain.shape
    open_list = PriorityQueue()
    open_list.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}

    while not open_list.empty():
        _, current = open_list.get()
        if current == goal:
            break
        for action in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + action[0], current[1] + action[1])
            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]:
                tentative_g_score = g_score[current] + env.calculate_energy_cost(current[0], current[1], neighbor[0], neighbor[1])
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    open_list.put((f_score[neighbor], neighbor))
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    total_energy_cost = g_score[goal]
    return path, total_energy_cost

astar_initial_path, astar_initial_energy_cost = a_star_path(env.start, env.goal, env.terrain, env.kd, env.kh, env.kr)
print(f"astar cost {astar_initial_energy_cost}")

# initial_path, initial_energy_cost = dijkstra_path(env.terrain, env.start, env.goal)
# print(f"dijkstra cost {initial_energy_cost}")



# Print energy costs for A* and PPO
print(f"Initial Path Energy Cost (A*): {astar_initial_energy_cost}")

initial_path_x = [point[1] for point in astar_initial_path]
initial_path_y = [point[0] for point in astar_initial_path]

ppo_path_x = [point[1] for point in ppo_path]
ppo_path_y = [point[0] for point in ppo_path]

# Visualize A* and PPO paths
plt.imshow(env.terrain, cmap='terrain', interpolation='bilinear')
plt.colorbar(label='Elevation')

# Plot A* path
plt.plot(initial_path_x, initial_path_y, color='orange', label='A* Path')
# Plot PPO path
plt.plot(ppo_path_x, ppo_path_y, color='blue', label='PPO Path')

# Mark start and goal points
plt.scatter(env.start[1], env.start[0], color='red', marker='o', label='Start')
plt.scatter(env.goal[1], env.goal[0], color='green', marker='x', label='Goal')

# Add legend
plt.legend()
plt.title('Comparison of Paths: A* vs PPO')
plt.show()