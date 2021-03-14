import random

from collections import namedtuple, deque

import numpy as np
import torch

from modules.utils.base import BaseMemory


class ReplayBuffer(BaseMemory):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, device, seed=42):
        """Initialize a ReplayBuffer object.

        Keywords:
            buffer_size (int): maximum size of buffer
            device (device): cuda or cpu to process tensors
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.device = device

        # Create memory object
        self.memory = deque(maxlen=buffer_size)

        # Create experience object
        self.experience = namedtuple(
            typename='Experience',
            field_names=['state', 'action', 'reward', 'next_state', 'done']
        )

    def add(self, data):
        """Add a new experience to memory."""
        state, action, reward, next_state, done = data
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=batch_size)

        states = torch\
            .from_numpy(
                np.vstack([e.state for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        actions = torch\
            .from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        rewards = torch\
            .from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        next_states = torch\
            .from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        dones = torch\
            .from_numpy(
                np.vstack([e.done for e in experiences if e is not None])
                .astype(np.uint8)
            )\
            .float()\
            .to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class HERReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_size, device, ratio_t_modified=0.85, seed=42):
        """Initialize a ReplayBuffer object.

        Keywords:
            buffer_size (int): maximum size of buffer
            device (device): cuda or cpu to process tensors
            seed (int): random seed

        """
        super(HERReplayBuffer, self).__init__(buffer_size, device)

        self.seed = random.seed(seed)
        self.device = device

        # Ratio of modifications to perform over all the batch experiences
        self.ratio_t_modified = ratio_t_modified

        # Create memory object
        self.memory = deque(maxlen=buffer_size)

        # Create experience object
        field_names = [
            'state',
            'action',
            'reward',
            'next_state',
            'done',
            'achieved_goal',
            'desired_goal'
        ]
        self.experience = namedtuple(
            typename='Experience',
            field_names=field_names
        )

    def add(self, data):
        """Add a new experience to memory."""
        state, action, reward, next_state, done, achieved_goal, desired_goal = data
        exp = self.experience(state, action, reward, next_state, done, achieved_goal, desired_goal)
        self.memory.append(exp)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=batch_size)

        states = torch \
            .from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ) \
            .float() \
            .to(self.device)

        actions = torch \
            .from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ) \
            .float() \
            .to(self.device)

        rewards = torch \
            .from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ) \
            .float() \
            .to(self.device)

        next_states = torch \
            .from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ) \
            .float() \
            .to(self.device)

        dones = torch \
            .from_numpy(
            np.vstack([e.done for e in experiences if e is not None])
                .astype(np.uint8)
        ) \
            .float() \
            .to(self.device)

        her_indexes = np.where(
            np.random.uniform(size=batch_size) < self.ratio_t_modified
        )

        # Flatten np.where
        her_indexes = her_indexes[0]

        goal_size = len(experiences[0].achieved_goal)

        for idx in her_indexes:
            # experiences[idx].next_state = experiences[idx].achieved_goal
            # experiences[idx].reward = self.reward_fn(
            #     experiences[idx].achieved_goal,
            #     experiences[idx].achieved_goal.copy(),
            #     experiences[idx].info
            # )
            states[idx] = torch.from_numpy(
                np.concatenate(
                    [experiences[idx].state[:-goal_size], experiences[idx].achieved_goal],
                    axis=0
                )
            ).to(self.device)
            next_states[idx] = torch.from_numpy(
                np.concatenate(
                    [experiences[idx].next_state[:-goal_size], experiences[idx].achieved_goal],
                    axis=0
                )
            ).to(self.device)
            rewards[idx] = torch.from_numpy(np.array([0])).to(self.device)
            dones[idx] = torch.from_numpy(np.array([1])).to(self.device)

        return states, actions, rewards, next_states, dones
