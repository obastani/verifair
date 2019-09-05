import numpy as np

class SampleException(Exception):
    pass

def qualified(b):
    if not b:
        raise SampleException()

def fairnessTarget(b):
    pass

def gaussian(mu, sigma):
    return np.random.normal(mu, np.sqrt(sigma))

def _check(vals):
    for i, val in enumerate(vals):
        if len(val) != 3 or val[0] != i or val[1] != i+1:
            raise Exception()

def step(vals):
    _check(vals)
    p = np.array([val[2] for val in vals])
    return np.random.choice(len(p), p=p)

class RejectionSampler:
    def __init__(self, sample_fn, flag):
        self.sample_fn = sample_fn
        self.flag = flag
        self.n_samples = 0

    def sample(self):
        while True:
            try:
                self.n_samples += 1
                return self.sample_fn(self.flag)
            except SampleException:
                pass

class MultiSampler:
    def __init__(self, sample_fn):
        self.sample_fn = sample_fn

    def sample(self, n_samples):
        samples = np.zeros([n_samples], dtype=np.int)
        for i in range(n_samples):
            samples[i] = self.sample_fn.sample()
        return samples
