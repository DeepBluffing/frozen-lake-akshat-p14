import gymnasium as gym
import numpy as np

def train_agent():
    # Initialize environment (is_slippery=False makes it deterministic for easier learning)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    # Hyperparameters
    num_episodes = 5000
    alpha = 0.5       # Learning rate
    gamma = 0.95      # Discount factor
    epsilon = 1.0     # Exploration rate
    min_epsilon = 0.01
    epsilon_decay = 0.995

    # Initialize Q-Table with zeros (16 states, 4 actions)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # ==========================================
            # TODO 1: Implement Epsilon-Greedy Action Selection [COMPLETED]
            # ==========================================
            if np.random.uniform(0, 1) < epsilon:
                # Explore: Choose a random legal direction
                action = env.action_space.sample()
            else:
                # Exploit: Choose the direction with the highest Q-value for this state
                action = np.argmax(q_table[state])
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ==========================================
            # TODO 2: The Bellman Equation Update [COMPLETED]
            # ==========================================
            # 1. Fetch the old score for this state-action pair
            old_q_value = q_table[state, action]
            
            # 2. Find the maximum possible Q-value out of all actions in the next state
            max_future_q = np.max(q_table[next_state])
            
            # 3. Apply the off-policy Temporal-Difference Bellman equation
            temporal_difference_target = reward + (gamma * max_future_q)
            new_q_value = old_q_value + alpha * (temporal_difference_target - old_q_value)
            
            # 4. Save the calculated value back into our Q-table grid matrix
            q_table[state, action] = new_q_value
            
            state = next_state
            
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table

if __name__ == "__main__":
    trained_q_table = train_agent()
    print("\nTraining complete! Final Q-Table:")
    print(trained_q_table)