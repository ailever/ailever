from ._environment import BaseEnvironment

from copy import deepcopy
import numpy as np
import torch


class NaiveEnv(BaseEnvironment):
    def __init__(self, actions, grid):
        self.memory = dict()
        self.s = None 
        self.nA = torch.tensor(len(actions))
        self.nS = torch.prod(torch.tensor(grid))
        self.isd = torch.ones(nS) / self.nS # Initial state distribution is uniform

        self.A = torch.arange(nA)        # self.A : Action Function Space
        self.S = torch.arange(nS)        # self.S : State Function Space
        self.P = torch.zeros(nA, nS, nS) # self.P : Transition Probability
        self.R = torch.zeros(nS, nA)     # self.R : Reward
        self.__update_PR()

    def __update_PR(self):
        # P[state][action] = (prob, next_state, reward, is_done)
        P = {}
        grid_array = np.empty(grid)
        it = np.nditer(grid_array, flags=['multi_index'])
        while not it.finished:
            state = it.iterindex
            P[state] = {} 
            for action in self.A:
                new_state = self._ProcessCore(state, action)
                P[state][action] = [(1.0, new_state, self.reward(state, action), self.termination_query(new_state))]
            it.iternext()

        for state in P.keys():
            for action in P[state].keys():
                prob, new_state, reward, done = P[state][action][0]
                self.P[action, state, new_state] = prob
                self.R[state, action] = reward
        
        self._gymP = P
    
    @property
    def gymP(self):
        return self._gymP

    def _ProcessCore(state, action=None):
        new_state = state + action
        if new_state < nS-1:
            return new_state
        else:
            new_state = torch.tensor(nS - 1)
            return new_state

    def termination_query(state):
        return state == 0 or state == (nS - 1)

    def reward(state, action=None):
        if termination_query(state) : return 0.0 
        else : return -1.0

    def step(self, action):
        cur_state = self.s
        next_state = self._ProcessCore(cur_state, action)
        reward = self.reward(cur_state, action)
        done = self.termination_query(next_state)
        return (next_state, reward, done, {"prob": 1.})

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
