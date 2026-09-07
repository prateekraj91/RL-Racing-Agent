import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=100000):

        self.capacity = capacity
        self.buffer = []
        self.position = 0
        # Number of leading entries that are never overwritten. 0 by default,
        # so behaviour is unchanged for every existing caller.
        self.protected = 0

    def protect_first(self, n):
        """Freeze the first `n` entries against circular-buffer eviction.

        Used for demonstration seeding: a run pushes more transitions than the
        buffer holds, so without this the demos are silently overwritten part
        way through training and the expert signal disappears exactly when the
        agent is still relying on it.
        """
        self.protected = min(n, len(self.buffer), self.capacity - 1)
        self.position = max(self.position, self.protected)
        if self.position >= self.capacity:
            self.position = self.protected
        return self.protected

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
            self.position = len(self.buffer) % self.capacity
            if self.position < self.protected:
                self.position = self.protected
            return

        self.buffer[self.position] = experience
        self.position += 1
        if self.position >= self.capacity:
            self.position = self.protected

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