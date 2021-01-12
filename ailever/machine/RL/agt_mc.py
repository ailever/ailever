from ._agent import BaseAgent

import torch

class MCAgent(BaseAgent):
    r"""
    Args:
        actions:
        grid:

    Examples:
        >>> from ailever.machine.RL import NaiveEnv
        >>> from ailever.machine.RL import MCAgent
        >>> ...
	>>> actions = {'a':0, 'b':1, 'c':2}
	>>> observation = {}
	>>> observation['grid'] = (3, 3)
	>>> env = NaiveEnv(actions, observation['grid'])
        >>> ...
	>>> agent = MCAgent(env)
        >>> ...
	>>> for epi_cnt in range(10):
	>>>     env.reset(); step = 0
	>>>     states = []; actions = []; rewards = []
	>>>     episode = (states, actions, rewards)
	>>>     while True:
	>>> 	    state = env.s
	>>> 	    action = agent.judge(state)
	>>> 	    next_state, reward, done, info = env.step(action); step += 1
        >>> ...
	>>> 	    states.append(state)
	>>> 	    actions.append(action)
	>>> 	    rewards.append(reward)
        >>> ...
	>>> 	    env.render(step)
        >>> ...
	>>> 	    if step == 1:
	>>> 	        observables = {'reward':reward, 'done':done}
	>>> 	        env.observe(step, epi_cnt, observables)
	>>> 	    if done : break
	>>>     agent.macro_update_Q(episode)
	>>> agent.update_policy()
	>>> env.memory

    Attributes:
        micro_update_Q: (*method*) **return**
        macro_update_Q: (*method*) **return**
        update_policy: (*method*) **return**
        judge: (*method*) **return**
        env: (*variable*)
        policy: (*variable*)
        gamma: (*variable*)
        V: (*variable*)
        Q: (*variable*)
    """

    def __init__(self, env=None):
        self.env = env
        self.policy = None
        self._setup_policy()

        self.gamma = 1.0
        self.V = None
        self.Q = None
        self._setup_Q()

    def _setup_policy(self):
        self.policy = torch.zeros(self.env.nS, self.env.nA)

    def _setup_Q(self):
        self.V = torch.zeros(self.env.nS)
        self.Q = torch.zeros(self.env.nS, self.env.nA)

    def micro_update_Q(self):
        pass

    def macro_update_Q(self, episode):
        V = {}; Q = {}
        V["s"] = self.V
        Q["s,a"] = self.Q

        states, actions, rewards= episode

        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        iter = zip(states, actions, rewards)
        G = 0; lr = 0.1
        for state, action, reward in iter:
            G += reward + self.gamma*G

            V["s"][state] += lr*(G - V["s"][state])
            Q["s,a"][state, action] += lr*(G - Q["s,a"][state, action])

        self.V = V["s"]
        self.Q = Q["s,a"]

    def update_policy(self):
        self.policy = self.Q

    def judge(self, state):
        action = self.policy[state].argmax(dim=0)
        return action
