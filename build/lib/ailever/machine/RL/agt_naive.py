from ._agent import BaseAgent

import torch

class NaiveAgent(BaseAgent):
    r"""
    Args:
        actions:
        grid:
    
    Examples:
        >>> from ailever.machine.RL import NaiveEnv
        >>> from ailever.machine.RL import NaiveAgent
        >>> ...
        >>> actions = {'a':0, 'b':1, 'c':2}
        >>> observation = {}
        >>> observation['grid'] = (3, 3)
        >>> ...
        >>> env = NaiveEnv(actions, observation['grid'])
        >>> env.reset()
        >>> ...
        >>> agent = NaiveAgent(env)
        >>> agent.macro_update_Q()
        >>> ...
        >>> for episode in range(1):
        >>>     step = 0
        >>>     while True:
        >>>         action = agent.judge(env.s)
        >>>         next_state, reward, done, info = env.step(action); step += 1
        >>>         env.render(step)
        >>>         if step == 1:
        >>>             observables = {'reward':reward, 'done':done}
        >>>             env.observe(step, episode, observables)
        >>>         if done : break
        >>> env.memory

    Attributes:
        micro_update_Q: (*method*) **return**
        macro_update_Q: (*method*) **return**
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
        self.policy = torch.ones((self.env.nS, self.env.nA)) / self.env.nA

    def _setup_Q(self):
        self.V = torch.zeros(self.env.nS)
        self.Q = self.env.R + self.gamma*torch.einsum("ijk,k->ji", [self.env.P, self.V])

    def micro_update_Q(self):
        pass

    def macro_update_Q(self):
        policy = {}; R = {}; P = {}; V = {}; Q = {}
        policy["s,a"] = self.policy

        P["a,s,s'"] = self.env.P
        R["s,a"] = self.env.R
        V["s"] = self.V

        R["s,a|pi"] = torch.einsum("ij,ij->ij", [policy["s,a"], R["s,a"]])
        P["a,s,s'|pi"] = torch.einsum("ji,ijk->ijk", [policy["s,a"], P["a,s,s'"]])

        step = -1
        gamma = 0.1
        while True:
            step += 1

            Q["s,a"] = R["s,a|pi"] + gamma*torch.einsum("ijk,k->ji", [P["a,s,s'|pi"], V["s"]])
            V["s|a=a*"] = Q["s,a"].max(dim=1)[0]
            self.V = V["s|a=a*"]

            Q["s,a"] = R["s,a|pi"] + gamma*torch.einsum("ijk,k->ji", [P["a,s,s'|pi"], V["s|a=a*"]])
            self.Q = Q["s,a"]

            policy["s,a*"] = torch.zeros_like(self.policy)
            policy["s,a*"][torch.arange(Q["s,a"].size(0)), Q["s,a"].argmax(dim=1)] = 1
            self.policy = policy["s,a*"]

            bellman_error = torch.linalg.norm(V["s|a=a*"] - V["s"])
            if bellman_error <= 0.001 : break
            else : V["s"] = V["s|a=a*"]

    def judge(self, state):
        action = self.policy[state].argmax(dim=0)
        return action



