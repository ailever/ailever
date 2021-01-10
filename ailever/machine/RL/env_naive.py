from _environment import BaseEnvironment

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

class NaiveEnv(BaseEnvironment):
    def __init__(self, actions, grid):
        self.grid = grid
        self.nA = torch.tensor(len(actions))
        self.nS = torch.prod(torch.tensor(grid))
        self.isd = torch.ones(self.nS) / self.nS # Initial state distribution is uniform

        self.s = torch.randint(0, self.nS, (1,)).squeeze()   # self.s  : current state
        self.termination_states = [0,self.nS-1]              # self.termination_states : termination states
        self.memory = dict()
        self.A = torch.arange(self.nA)                  # self.A : Action Function Space
        self.S = torch.arange(self.nS)                  # self.S : State Function Space
        self.P = torch.zeros(self.nA, self.nS, self.nS) # self.P : Transition Probability
        self.R = torch.zeros(self.nS, self.nA)          # self.R : Reward
        self._update_PR()
        self._update_gymP()

    def _update_PR(self):
        # self.P
        for action in self.A:
            for state in self.S:
                self.P[action,state,:] = self.isd

        # self.R
        for termination_state in self.termination_states:
            self.R[termination_state, :] = 1

    def _update_gymP(self):
        # P[state][action] = (prob, next_state, reward, is_done)
        P = {}
        grid_array = np.empty(self.grid)
        it = np.nditer(grid_array, flags=['multi_index'])
        while not it.finished:
            state = it.iterindex
            P[state] = {} 
            for action in self.A:
                for new_state in self.S:
                    prob = float(self.P[action, state, new_state])
                    reward = float(self.R[state, action])
                    P[state][action] = [(prob, new_state, reward, self.termination_query(new_state))]
            it.iternext()
        self._gymP = P
        
        """
        for state in P.keys():
            for action in P[state].keys():
                for next_step in P[state][action]:
                    prob, new_state, reward, done = next_step
                    self.P[action, state, new_state] = prob
                    self.R[state, action] = reward
        """

    @property
    def gymP(self):
        return self._gymP


    def _ProcessCore(self, state, action=False):
        transition = self.P[int(action), state]
        samples = self.sampler(probs=transition)
        self.s = samples.argmax()
        new_state = self.s
        return new_state, transition[new_state]

    def termination_query(self, state):
        if state in self.termination_states:
            return True
        else:
            return False

    def reward(self, state, action=False):
        return self.R[state, action]

    def sampler(self, probs=[0.1, 0.9], size=1):
        total_count = 1
        size = torch.Size([size])
        probs = torch.tensor(probs)
        samples = Multinomial(total_count=total_count, probs=probs).sample(sample_shape=size).squeeze()
        return samples

    def step(self, action):
        cur_state = self.s
        next_state, prob = self._ProcessCore(cur_state, action)
        reward = self.reward(cur_state, action)
        done = self.termination_query(next_state)
        return (next_state, reward, done, {"prob": prob})

    def render(self, step_cnt, mode=None, verbose=False):
        if verbose : return
        print(f'\n[ STEP : {step_cnt} ]')
        print(f'- Current State : {self.s}')

    def observe(self, step_cnt, observables=dict()):
        assert isinstance(observables, dict), 'Your observables must be dict type.'

        observations = {}
        observations['Current State'] = deepcopy(self.s)
        
        for key, observable in observables.items():
            observations[key] = deepcopy(observable)

        self.memory[step_cnt] = observations


