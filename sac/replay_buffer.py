import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=100000):

        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):

        experience = (
            state,
            action,
            reward,
            next_state,
            done,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        indices = np.random.randint(
            0,
            len(self.buffer),
            size=batch_size,
        )

        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):

        return len(self.buffer)