from environment import SimpleKSPEnv
from agents import QLearningAgentANN

# 1. Create your Gym environment
ske = SimpleKSPEnv()

# 2. Create an instance of the QLearningAgent
agent = QLearningAgentANN(
    env=ske,
    learning_rate=0.001,  # Adjust learning rate
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.996,
    min_epsilon=0.01,
)  # Adjust number of bins for state variables

# 3. Training loop
for episode in range(1000):  # Adjust number of episodes
    state = ske.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")