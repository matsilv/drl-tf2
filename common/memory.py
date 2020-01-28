# Author: Mattia Silvestri

from collections import deque
import random


# replay memory as ring buffer
class ReplayExperienceBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = deque(maxlen=maxlen)

    def insert(self, entry):
        self.memory.append(entry)

    def get_random_batch(self, batch_size):
        return random.sample(list(self.memory), batch_size)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

