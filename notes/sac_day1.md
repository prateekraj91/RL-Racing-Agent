Abstract

Problems in existing RL
- Needs millions of samples
- Very sensitive to hyperparameters

SAC
- Off-policy Actor-Critic
- Uses Maximum Entropy RL
- Maximizes reward + entropy
- Uses stochastic policy

Goal
- Better exploration
- Better stability
- Better sample efficiency

Section 1: Introduction

Problems with existing RL
- Needs huge amounts of data (poor sample efficiency).
- Very sensitive to hyperparameters.
- Difficult to use reliably in real-world tasks.

On-policy vs Off-policy
- On-policy: learns only from newly collected experience.
- Off-policy: reuses old experience from a replay buffer.
- Off-policy is more sample efficient but harder to stabilize.

DDPG
- Off-policy and sample efficient.
- Very brittle and sensitive to hyperparameters.

Maximum Entropy RL
- Optimizes reward + entropy.
- Encourages exploration.
- Produces more robust policies.

SAC's Goal
- Combine:
  - Off-policy learning
  - Actor-Critic
  - Maximum Entropy
- Achieve better stability and sample efficiency.

Section 2: Related Work

SAC combines three ideas
- Actor-Critic architecture
- Off-policy learning
- Maximum Entropy objective

Actor-Critic
- Critic evaluates the current policy.
- Actor improves the policy.
- Repeat until the policy gets better.

Previous Actor-Critic methods
- Used entropy mainly as a regularizer.
- SAC makes entropy part of the optimization objective.

DDPG
- Off-policy and sample efficient.
- Uses a deterministic policy.
- Difficult to stabilize.

SVG(0)
- Does not optimize the maximum entropy objective.
- Does not use a separate value network.

Previous Maximum Entropy methods
- Improved exploration.
- Did not consistently outperform DDPG.

SAC
- Combines off-policy learning with a stochastic actor.
- More stable.
- Better sample efficiency.

Section 3: Preliminaries

MDP
- S = States
- A = Actions
- p = Transition function
- r = Reward function

Policy
- Maps states to actions.

Standard RL
- Maximizes expected reward.

Maximum Entropy RL
- Maximizes reward + entropy.
- Encourages exploration.
- Produces more robust policies.

Alpha (α)
- Controls the importance of entropy.
- Small α → more reward-focused.
- Large α → more exploration.

Advantages of Maximum Entropy
- Better exploration.
- Can represent multiple good actions.
- Faster and more stable learning.

SAC vs TD3

Similarities
- Both are off-policy algorithms.
- Both use a replay buffer.
- Both are Actor-Critic methods.
- Both work well in continuous action spaces.

TD3
- Uses a deterministic policy.
- Optimizes only expected reward.
- Exploration comes from adding external noise.
- More sensitive to hyperparameters.
- Can be unstable during training.

SAC
- Uses a stochastic policy.
- Optimizes reward + entropy.
- Exploration is built into the objective.
- Better exploration.
- More stable training.
- Better sample efficiency.

Main Idea

TD3:
Reward → Best action

SAC:
Reward + Entropy → Best distribution of actions

Why SAC was proposed

To combine:
- Off-policy learning
- Actor-Critic architecture
- Maximum Entropy RL

Result:
- Faster learning
- Better exploration
- More stable training

Section 4: Soft Policy Iteration

Soft Policy Iteration

Goal:
- Improve the policy while maximizing both reward and entropy.

Two steps:

1. Policy Evaluation
- Estimate how good the current stochastic policy is.
- Includes both expected reward and entropy.

2. Policy Improvement
- Update the actor toward actions with higher soft Q-values.
- The improved policy remains stochastic.

Repeat:
Policy evaluation → Policy improvement → better policy

Section 5: Soft Actor-Critic

SAC Core Components

1. Actor
- Stochastic policy.
- Outputs a probability distribution over actions.
- Learns which actions are useful while maintaining exploration.

2. Critic
- Estimates Q(s, a).
- SAC uses two Q-functions to reduce overestimation bias.

3. Replay Buffer
- Stores past transitions:
  (state, action, reward, next_state, done)
- Allows off-policy learning.
- Experience can be reused multiple times.

4. Entropy Temperature α
- Controls reward vs exploration.
- Higher α → stronger exploration.
- Lower α → more reward-focused behavior.

5. Target Networks
- Stabilize critic learning.
- Slowly track the learned Q-networks.

SAC Training Loop

Environment
    ↓
state
    ↓
Actor
    ↓
action
    ↓
Environment
    ↓
reward + next state
    ↓
Replay Buffer
    ↓
Critic update
    ↓
Actor update
    ↓
Temperature α update
    ↓
Repeat

Your Racing Environment
        ↓
state
        ↓
SAC Actor
        ↓
steering / throttle action
        ↓
Racing Environment
        ↓
reward
        ↓
Replay Buffer
        ↓
SAC learns

Reward Shaping

Problem

- Sparse rewards can make RL difficult to learn.
- We can add intermediate rewards to provide more learning signal.
- Arbitrary reward shaping can change the optimal policy.
- The agent may learn to maximize the shaping reward instead of the actual task objective.

Example in racing

Bad shaping:

- +1 for moving forward
- +1 for high speed
- +1 for staying near the center

Potential problem:

- The car may learn to drive fast or stay near the center without actually finishing the race.

Potential-Based Reward Shaping

Ng et al. (1999) showed that policy invariance can be preserved using potential-based shaping.

F(s, s') = γΦ(s') - Φ(s)

Where:

- Φ(s) = potential function representing the desirability/progress of state s.
- γ = discount factor.
- F(s,s') = shaping reward.

Shaped reward:

R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)

For the racing agent:

- Φ(s) could represent progress along the track.
- Moving forward → positive shaping reward.
- Moving backward → negative shaping reward.

Key takeaway

- Do not add arbitrary rewards without considering how they change the objective.
- Potential-based shaping provides dense feedback while preserving the optimal policy under the theorem's assumptions.