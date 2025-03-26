"""
Gambler’s Problem

A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money. On each flip, the gam- bler must decide what portion of his capital to stake, in integer numbers of dollars. 

This problem can be formulated as an undiscounted, episodic, finite MDP. The state is the gambler’s capital, $s \in \{1,2,...,99\}$ and the actions are stakes, $a \in \{0,1, \ldots, \min(s,100 - s)}. The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1.


Si bien el problema se menciona como un problema simple, en este artículo

https://openreview.net/pdf?id=HyxnMyBKwB

se muestra que existen multiples soluciones y que no es nunca tan obvio como parecería.
"""

from MDPs import MDP, iteracion_valor
from matplotlib import pyplot as plt

class Gambler(MDP):
    """
    Clase que representa un MDP para el problema del apostador.
    
    """
    
    def __init__(self, gama=0.99, meta=100, ph=0.5):
        self.gama = gama
        self.meta = meta
        self.ph = ph
        self.estados = tuple(range(0, meta + 2))
    
    def acciones_legales(self, s):
        if s == 0 or s == self.meta + 1:
            return []
        elif s == self.meta:
            return [1]
        return range(1, min(s, self.meta - s) + 1)
    
    def recompensa(self, s, a, s_):
        return 1 if s == self.meta else 0
    
    def prob_transicion(self, s, a, s_):
        if s == 0 or s == self.meta + 1:
            return 0
        if s == self.meta:
            return 1 if s_ == self.meta + 1 else 0
        return self.ph if s_ == s + a else 1.0 - self.ph if s_ == s - a else 0
    
    def es_terminal(self, s):
        return s == 0 or s == self.meta + 1

mdp = Gambler(gama=1, ph=0.5)    
pi_star, V_star = iteracion_valor(
    mdp, 
    epsilon=1e-6, max_iter=10_000, ver_V=True, debug=True
)

plt.plot(range(1, 100), [pi_star[s] for s in range(1, 100)], '*')
plt.xlabel("Capital")
plt.ylabel("Apuesta")
plt.show()
