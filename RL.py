"""
Modulo com modelo de simulaciñon de MDPs y algoritmos de RL

"""

from abc import ABCMeta, abstractmethod
from random import choice, random

class MDPsim(metaclass=ABCMeta):
    def __init__(self, estados, gama):
        self.estados = estados
        self.gama = gama
        
    @abstractmethod
    def estado_inicial(self):
        """
        Devuelve el estado inicial.
        
        """
        raise NotImplementedError("Estado inicial no implementado")
    
    @abstractmethod
    def acciones_legales(self, s):
        """
        Devuelve una lista con las acciones legales en el estado s.
        
        """
        raise NotImplementedError("Acciones legales no implementada")
    
    @abstractmethod
    def recompensa(self, s, a, s_):
        """
        Devuelve la recompensa de la transición s, a, s'.
        
        """
        raise NotImplementedError("Recompensa no implementada")
    
    @abstractmethod
    def transicion(self, s, a):
        """
        Devuelve el estado s'
        
        """
        raise NotImplementedError("Transición no implementada")
    
    def es_terminal(self, s):
        """
        Devuelve True si el estado s es terminal.
        
        """
        return False

def TD0(mdp, politica, alfa, n_ep, n_iter):
    """
    Algoritmo de TD(0) para estimar la función de valor de un MDP.
    
    Parámetros:
        mdp: objeto de la clase MDP
        politica: diccionario que devuelve la acción en un estado
        alfa: tasa de aprendizaje
        n_ep: número máximo de episodios
        n_iter: número máximo de iteraciones por episodio
    
    """
    V = {s: 0 for s in mdp.estados}
    
    for _ in range(n_ep):
        s = mdp.estado_inicial()
        for _ in range(n_iter):
            a = politica[s]
            s_ = mdp.transicion(s, a)
            V[s] += alfa * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_] - V[s])
            if mdp.es_terminal(s_):
                break
            s = s_  
    return V

def politica_e_greedy(Q, s, acciones, epsilon):
    """
    Política epsilon-greedy.
    
    Parámetros:
        Q: diccionario con la función de valor Q
        s: estado
        acciones: lista con las acciones legales
        epsilon: probabilidad de exploración
    
    """
    if random() < epsilon:
        return choice(acciones)
    else:
        return max(acciones, key=lambda a: Q[(s, a)])

def SARSA(mdp, epsilon, alfa, n_ep, n_iter):
    """
    Algoritmo SARSA para estimar la función de valor de un MDP.
    
    Parámetros:
        mdp: objeto de la clase MDP
        epsilon: probabilidad de exploración
        alfa: tasa de aprendizaje
        n_ep: número de episodios
        n_iter: número de iteraciones
    
    """
    Q = {(s, a): random() 
         for s in mdp.estados if not mdp.es_terminal(s) 
         for a in mdp.acciones_legales(s)}
        
    for _ in range(n_ep):
        s = mdp.estado_inicial()
        a = politica_e_greedy(Q, s, mdp.acciones_legales(s), epsilon)
        for _ in range(n_iter):
            s_ = mdp.transicion(s, a)
            r = mdp.recompensa(s, a, s_)
            if mdp.es_terminal(s_):
                Q[(s, a)] += alfa * (r - Q[(s, a)])
                break
            a_ = politica_e_greedy(Q, s_, mdp.acciones_legales(s_), epsilon)
            Q[(s, a)] += alfa * (r + mdp.gama * Q[(s_, a_)] - Q[(s, a)])
            s, a = s_, a_       
    return Q

def Q_learning(mdp, epsilon, alfa, n_ep, n_iter):
    """
    Algoritmo Q-learning para estimar la función de valor de un MDP.
    
    Parámetros:
        mdp: objeto de la clase MDP
        epsilon: probabilidad de exploración
        alfa: tasa de aprendizaje
        n_ep: número de episodios
        n_iter: número de iteraciones
    
    """
    Q = {(s, a): 0 
         for s in mdp.estados if not mdp.es_terminal(s)
         for a in mdp.acciones_legales(s)}
    
    for _ in range(n_ep):
        s = mdp.estado_inicial()
        for _ in range(n_iter):
            a = politica_e_greedy(Q, s, mdp.acciones_legales(s), epsilon)
            s_ = mdp.transicion(s, a)
            r = mdp.recompensa(s, a, s_)
            if mdp.es_terminal(s_):
                Q[(s, a)] += alfa * (r - Q[(s, a)])
                break
            Q[(s, a)] += alfa * (
                r 
                + mdp.gama * max(Q[(s_, a_)] for a_ in mdp.acciones_legales(s_)) 
                - Q[(s, a)])
            s = s_
    return Q