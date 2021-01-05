import numpy as np
from gym.envs.toy_text import discrete

class MDP(discrete.DiscreteEnv):
    r"""
    Args:
        actions:
        grid:
    
    .. note::
        MDP = <S,A,P,R>
            - S : State Space
            - A : Action Space
            - P : Transition Probability
            - R : Reward

    Attributes:
        reset: (*method*) **return**
        step: (*method*) **return** 
        render: (*method*) **return** 
        observation_space: (*variable*) Observation Space
        action_space: (*variable*) Action Space
        s: (*variable*) Current State
        PTensor: (*variable*) Transition Probability
        RTensor: (*variable*) Reward
    
    Examples:
        >>> from ailever.machine.RL import MDP
        >>> ...
	>>> action = {}
        >>> actions['a'] = 0
	>>> actions['b'] = 1
	>>> actions['c'] = 2
	>>> actions['d'] = 3
	>>> observation = {}
	>>> observation['grid'] = (3, 3)
        >>> ...
	>>> mdp = MDP(actions, observation['grid'])
        >>> ...
        >>> step = 0
        >>> while:
        >>>     action = np.random.randint(low=0, high=4)
        >>>     mdp.render(step)
        >>>     next_state, reward, done, info = mdp.step(action); step += 1
        >>>     if done : break

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, actions, grid):
        nA = len(actions)
        nS = np.prod(grid)
        isd = np.ones(nS) / nS                      # Initial state distribution is uniform

        self.A = np.arange(nA)                      # self.A : Action Space
        self.S = np.arange(nS)                      # self.S : State Space
        self.PTensor = np.zeros(shape=(nA, nS, nS)) # self.PTensor : Transition Probability
        self.RTensor = np.zeros(shape=(nS, nA))     # self.RTensor : Reward

        # P[state][action] = [(probabilty, ProcessCore(state, action), reward(state, action), termination_query(ProcessCore(state, action)))]
        P = self.update(actions, grid, nS)
        super(MDP, self).__init__(nS, nA, P, isd)

    def update(self, actions, grid, nS):
        def ProcessCore(state, action=None):
            new_state = state + action
            if new_state < nS-1:
                return new_state
            else:
                new_state = nS-1
                return new_state

        def termination_query(state):
            return state == 0 or state == (nS - 1)

        def reward(state, action=None):
            if termination_query(state) : return 0.0 
            else : return -1.0

        # P[state][action] = (prob, next_state, reward, is_done)
        P = {}
        grid_array = np.empty(grid)
        it = np.nditer(grid_array, flags=['multi_index'])
        while not it.finished:
            state = it.iterindex
            P[state] = {} 
            for action in self.A:
                new_state = ProcessCore(state, action)
                P[state][action] = [(1.0, new_state, reward(state, action), termination_query(new_state))]
            it.iternext()

        for state in P.keys():
            for action in P[state].keys():
                prob, new_state, reward, done = P[state][action][0]
                self.PTensor[action, state, new_state] = prob
                self.RTensor[state, action] = reward

        return P

    def render(self, step, mode='human', verbose=False):
        if verbose : return
        print(f'\n[ STEP : {step} ]')
        print(f'- Current State : {self.s}')



