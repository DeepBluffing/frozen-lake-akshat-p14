[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QH74iBUg)
# Deep Bluffing: Assignment 2

You will build a Q-Learning algorithm. You have been provided with a Python starter files. Your task is to complete the mathematical updates within the `TODO` blocks.

> **Note:** Do not modify the function signatures or the boilerplate environment setup. Focus solely on translating the Bellman equations into Python logic.


## Task: Q-Learning on a Frozen Lake (Model-Free)

**File:** `frozen_lake_agent.py`

The agent is placed on a frozen lake and must learn the environment dynamics purely through trial and error. Because the agent does not know $\mathcal{P}_{ss'}^a$, it cannot calculate $V(s)$. It must learn the Action-Value function $Q(s,a)$ directly. You will guide an agent across a frozen lake to reach a treasure. After finishing the code, run `visualize_agent.py` to see it in action!

**Your Task:**

1. **Epsilon-Greedy Exploration:** Implement the logic to balance exploration (taking a random action) and exploitation (taking the best known action).
2. **Temporal-Difference Update:** Update the Q-Table using the off-policy Bellman equation:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[R+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$$


## How to Submit
Simply push your code to the `main` branch. 
