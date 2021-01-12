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
        >>> ...
        >>> agent = NaiveAgent(env)
        >>> agent.macro_update_Q()
        >>> ...
        >>> for epi_cnt in range(1):
        >>>     env.reset(); step = 0
        >>>     while True:
        >>>         action = agent.judge(env.s)
        >>>         next_state, reward, done, info = env.step(action); step += 1
        >>>         env.render(step)
        >>>         if step == 1:
        >>>             observables = {'reward':reward, 'done':done}
        >>>             env.observe(step, epi_cnt, observables)
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
        policy = {}; G = {}; R = {}; P = {}; V = {}; Q = {}
        policy["s,a"] = self.policy

        P["a,s,s'"] = self.env.P
        R["s,a"] = self.env.R
        V["s'"] = self.V
        gamma = 0.1

        step = -1
        while True:
            # policy evaluation
            while True:
                G["s,a"] = R["s,a"] + gamma*torch.einsum("ijk,k->ji", [P["a,s,s'"], V["s'"]])
                V["s"] = torch.einsum("ij,ij->i", [policy["s,a"], G["s,a"]])
                self.V = V["s"]

                bellman_error = torch.linalg.norm(V["s"] - V["s'"])
                if bellman_error <= 0.001 : break
                else : V["s'"] = V["s"]

            # policy improvement
            G["s,a"] = R["s,a"] + gamma*torch.einsum("ijk,k->ji", [P["a,s,s'"], V["s'"]])
            Q["s,a"] = torch.einsum("ij,ij->ij", [policy["s,a"], G["s,a"]])
            V["s'|a=a*"] = Q["s,a"].max(dim=1)[0]  # optimal policy theorem
            self.V = V["s'|a=a*"]

            G["s,a"] = R["s,a"] + gamma*torch.einsum("ijk,k->ji", [P["a,s,s'"], V["s'|a=a*"]])
            Q["s,a"] = torch.einsum("ij,ij->ij", [policy["s,a"], G["s,a"]])
            self.Q = Q["s,a"]
            
            policy["s,a*"] = torch.zeros_like(self.policy)
            policy["s,a*"][torch.arange(Q["s,a"].size(0)), Q["s,a"].argmax(dim=1)] = 1 # policy improvement theorem
            self.policy = policy["s,a*"]

            policy_gap = torch.linalg.norm(policy["s,a*"] - policy["s,a"])
            if policy_gap <= 0.001 : break
            else : policy["s,a"] = policy["s,a*"]

    def judge(self, state):
        action = self.policy[state].argmax(dim=0)
        return action



