"""
Módulo que contiene ejemplos del camion mágico estocástico.

"""

from MDPs import MDP, iteracion_politica, iteracion_valor

class CamionMagicoProb(MDP):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 2))
    
    def acciones_legales(self, s):
        return ['caminar', 'usar_camion']
    
    def recompensa(self, s, a, s_):
        return (
            -9  if s_ > self.meta else
             0  if s_ == self.meta else
            -1  if a == 'caminar' else -2   
        ) 
        
    def prob_transicion(self, s, a, s_):
        if s >= self.meta:
            return 1 if s_ == s else 0
        elif a == 'caminar':
            return 1 if s_ == min(s + 1, self.meta + 1) else 0
        elif a == 'usar_camion':
            return (self.rho if s_ == min(self.meta + 1, 2*s) else 
                    1 - self.rho if s_ == s else 0)
                
    def es_terminal(self, s):
        return False
    
for rho in [0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99]:
    print(f"Para rho = {rho}")
    pi_star = iteracion_valor(CamionMagicoProb(0.9, rho, 145))
    print(f"Los tramos donde se debe usar el camión son:")
    print([s for s in pi_star if pi_star[s] == 'usar_camion'])
    print("-"*50)
