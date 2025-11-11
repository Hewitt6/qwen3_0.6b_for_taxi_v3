# import gymnasium as gym
# import numpy as np
# import random
# from collections import defaultdict

# class QLearningAgent:
#     def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
#         self.env = env
#         self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min

#     def choose_action(self, state):
#         if random.uniform(0, 1) < self.epsilon:
#             return self.env.action_space.sample()
#         else:
#             return np.argmax(self.q_table[state])

#     def learn(self, state, action, reward, next_state):
#         old_value = self.q_table[state][action]
#         next_max = np.max(self.q_table[next_state])
#         new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
#         self.q_table[state][action] = new_value

#     def decode_state(self, state):
#         """Convert numerical state to human-readable text description"""
#         # Manual decoding for Taxi-v3 state
#         # Taxi-v3 state space: 500 states = 5x5 grid × 5 passenger locations × 4 destinations
#         # State encoding: (taxi_row * 100) + (taxi_col * 20) + (passenger_location * 4) + destination
        
#         taxi_row = state // 100
#         taxi_col = (state % 100) // 20
#         pass_loc = (state % 20) // 4
#         dest_idx = state % 4
        
#         locations = ["Red", "Green", "Yellow", "Blue", "in taxi"]
        
#         state_description = {
#             "taxi_position": f"({taxi_row}, {taxi_col})",
#             "passenger_location": locations[pass_loc],
#             "destination": locations[dest_idx],
#             "raw_state": int(state)
#         }
        
#         return state_description

#     def train(self, episodes):
#         successful_episodes = []
#         for i in range(episodes):
#             state, info = self.env.reset()
#             done = False
#             truncated = False
#             episode_data = []
#             total_reward = 0

#             while not done and not truncated:
#                 action = self.choose_action(state)
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 self.learn(state, action, reward, next_state)
                
#                 decoded_state = self.decode_state(state)
#                 episode_data.append({
#                     "state": decoded_state,
#                     "action": int(action),
#                     "reward": float(reward)
#                 })
                
#                 state = next_state
#                 total_reward += reward

#             if reward == 20:  # Successfully dropped off the passenger
#                 successful_episodes.append({
#                     "episode": int(i),
#                     "total_reward": float(total_reward),
#                     "steps": episode_data
#                 })

#             if self.epsilon > self.epsilon_min:
#                 self.epsilon *= self.epsilon_decay
            
#             if (i + 1) % 100 == 0:
#                 print(f"Episode {i + 1}/{episodes} completed.")

#         return successful_episodes

# def solve_taxi_v3_and_collect_data():
#     env = gym.make("Taxi-v3")
#     agent = QLearningAgent(env)
#     successful_episodes = agent.train(episodes=50)
#     env.close()
#     return successful_episodes

# if __name__ == "__main__":
#     collected_data = solve_taxi_v3_and_collect_data()
#     print(f"Collected {len(collected_data)} successful episodes.")
#     if collected_data:
#         print("\nExample successful trajectory:")
#         first_episode = collected_data[0]
#         print(f"Total reward: {first_episode['total_reward']}")
#         print(f"Steps: {len(first_episode['steps'])}")
import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            # 确保 state 是可哈希的，这里直接使用 state 作为 key
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def decode_state(self, state):
        """Convert numerical state to human-readable text description"""
        # Manual decoding for Taxi-v3 state
        taxi_row = state // 100
        taxi_col = (state % 100) // 20
        pass_loc = (state % 20) // 4
        dest_idx = state % 4
        
        locations = ["Red", "Green", "Yellow", "Blue", "in taxi"]
        
        state_description = {
            "taxi_position": f"({taxi_row}, {taxi_col})",
            "passenger_location": locations[pass_loc],
            "destination": locations[dest_idx],
            "raw_state": int(state)
        }
        
        return state_description

    def train(self, episodes):
        successful_episodes = []
        reward_history = []  # 🌟 修正 1: 添加奖励历史列表
        
        for i in range(episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_data = []
            total_reward = 0

            while not done and not truncated:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.learn(state, action, reward, next_state)
                
                decoded_state = self.decode_state(state)
                episode_data.append({
                    "state": decoded_state,
                    "action": int(action),
                    "reward": float(reward)
                })
                
                state = next_state
                total_reward += reward

            # 🌟 修正 2: 记录每回合的总奖励
            reward_history.append(total_reward) 

            if reward == 20:  # Successfully dropped off the passenger
                successful_episodes.append({
                    "episode": int(i),
                    "total_reward": float(total_reward),
                    "steps": episode_data
                })

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            if (i + 1) % 1000 == 0: # 减少打印频率，避免 Notebook 卡顿
                print(f"Episode {i + 1}/{episodes} completed. Epsilon: {self.epsilon:.4f}")

        # 🌟 修正 3: 返回两个值
        return successful_episodes, reward_history

def solve_taxi_v3_and_collect_data():
    # 调整训练回合数以提高成功率和数据量
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(env, episodes=5000) # 这里可能需要修改 train 的调用方式以兼容新的 train 方法签名
    successful_episodes, _ = agent.train(episodes=5000) # 假设 train 接收 episodes
    env.close()
    return successful_episodes

if __name__ == "__main__":
    collected_data, _ = QLearningAgent(gym.make("Taxi-v3")).train(episodes=100)
    print(f"Collected {len(collected_data)} successful episodes.")
    if collected_data:
        print("\nExample successful trajectory:")
        first_episode = collected_data[0]
        print(f"Total reward: {first_episode['total_reward']}")
        print(f"Steps: {len(first_episode['steps'])}")