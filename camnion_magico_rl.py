"""
El camión mágico, pero ahora por simulación

"""

from RL import MDPsim, SARSA, Q_learning
from random import random, randint

class CamionMagico(MDPsim):
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
    
    def estado_inicial(self):
        #return randint(1, self.meta // 2 + 1)
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s >= self.meta:
            return []
        return ['caminar', 'usar_camion']
    
    def recompensa(self, s, a, s_):
        return (
            -100  if s_ > self.meta else
             100  if s_ == self.meta else
            -1  if a == 'caminar' else -2   
        ) 
        
    def transicion(self, s, a):
        if a == 'caminar':
            return min(s + 1, self.meta + 1)
        elif a == 'usar_camion':
            return min(self.meta + 1, 2*s) if random() < self.rho else s
        
    def es_terminal(self, s):
        return s >= self.meta

mdp_sim = CamionMagico(
    gama=0.999, rho=0.9, meta=145
)
    
Q_sarsa = SARSA(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=50
)
pi_s = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_sarsa[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

Q_ql = Q_learning(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=1000
)
pi_ql = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_ql[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

print(f"Los tramos donde se debe usar el camión segun SARSA son:")
print([s for s in pi_s if pi_s[s] == 'usar_camion'])
print("-"*50)
print(f"Los tramos donde se debe usar el camión segun Qlearning son:")
print([s for s in pi_ql if pi_ql[s] == 'usar_camion'])
print("-"*50)


"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************
1.- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
    Si rho>0.5, el agente arriesga más porque es más probable ganar.
    Si rho<0.5, apuesta menos porque perder es más fácil.
    Si rho=0.5, el agente no arriesga nada porque no tiene sentido.
    Si rho=0, el agente no avanza nada porque no tiene sentido.

2.- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
    Si gamma es muy alto, el agente arriesga más porque le importa más el futuro.
    Si gamma es muy bajo, el agente arriesga menos porque le importa más el presente.
    Si gamma=0, el agente no arriesga nada porque no tiene sentido.

3.- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
    SARSA aprende según lo que realmente hace el agente.
    Q-learning aprende como si siempre tomara la mejor acción.
    SARSA suele ser más conservador y Q-learning más agresivo.

4.- ¿Cambia mucho el resultado cambiando los valores de recompensa?
    Sí, puede cambiar dependiendo de cómo se premien las acciones.
    Si la recompensa solo se da al llegar a la meta, el agente busca llegar ahí rápido.
    Si das más recompensas en el camino, puede que el agente prefiera otras rutas.
    Depende de qué tanto premies cada parte del juego.

6.- ¿Qué pasa si el costo de usar el camión es mayor que el de caminar?
    El juego sería más difícil.
    El agente empieza con menos dinero y más lejos de la meta.
    Entonces tiene más riesgo de perder, y probablemente apueste menos o le tome más tiempo aprender a ganar.
**********************************************************************************

"""