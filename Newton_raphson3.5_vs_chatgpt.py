# -*- coding: utf-8 -*-
"""
Ponto Fixo vs Newton-Raphson para magnetização autoconsistente
Com a MESMA definição de Bef em ambos os métodos
"""

import numpy as np  # Biblioteca para cálculos numéricos com arrays
import matplotlib.pyplot as plt  # Biblioteca para criar gráficos e visualizações
import time   # Biblioteca para medir tempo de execução

# -----------------------------
# Constantes físicas (em meV)
# -----------------------------
MIb = 0.0578838   # meV/T  (magnéton de Bohr - momento magnético fundamental)
Kb  = 0.086       # meV/K  (constante de Boltzmann - relaciona temperatura com energia)

# Parâmetros do problema físico (sistema magnético)
g   = 2.0         # Fator g de Landé (razão giromagnética)
j   = 7.0/2.0     # Número quântico de momento angular total (spin 7/2)
b   = 0.0         # Campo magnético externo aplicado (zero neste caso)
lam = 2.77        # Parâmetro de acoplamento molecular (constante de campo molecular)

# Grade de temperaturas para calcular a magnetização
# Cria array de 1K a 300K com passo de 1K
T_grid = np.arange(1., 300.1, 1.)

# -----------------------------
# Brillouin e auxiliares
# -----------------------------
def bril(x, j):
    
    """
    Função de Brillouin - descreve a magnetização de um sistema de spins
    x: argumento adimensional = (g*MIb*j*Bef)/(Kb*T)
    j: número quântico de momento angular
    Retorna: valor da função de Brillouin para dado x e j
    """
    A = (2.0*j + 1.0)/(2.0*j)  # Parâmetro A da função Brillouin
    B = 1.0/(2.0*j)            # Parâmetro B da função Brillouin
    return A/np.tanh(A*x) - B/np.tanh(B*x)  # Fórmula padrão da função Brillouin
    
def dbril_dx(x, j):
    """
    Derivada analítica da função de Brillouin em relação a x
    Necessária para o método de Newton-Raphson
    """
    A = (2.0*j + 1.0)/(2.0*j)
    B = 1.0/(2.0*j)
    # Derivada calculada analiticamente
    return -(A**2)/np.sinh(A*x)**2 + (B**2)/np.sinh(B*x)**2

def x_arg(m, T, g, j, lam, b):
    """
    Calcula o argumento x para a função de Brillouin
    m: magnetização atual
    T: temperatura
    Retorna: valor de x = (g*MIb*j*Bef)/(Kb*T)
    """
    # Campo efetivo = campo molecular (proporcional a m) + campo externo
    Bef = m * (lam / (g**2 * MIb)) + b
    # Argumento adimensional para função Brillouin
    return (g * MIb * j * Bef) / (Kb * T)

# -----------------------------
# f(m), f'(m) e g(m)=f(m)-m
# -----------------------------
def f_of_m(m, T):
    """
    Função de autoconsistência para magnetização
    Representa o lado direito da equação m = f(m)
    """
    x = x_arg(m, T, g, j, lam, b)  # Calcula argumento
    return g * j * bril(x, j)       # Retorna nova estimativa de magnetização

def fprime_of_m(m, T):
    """
    Derivada de f(m) em relação a m (usando regra da cadeia)
    Necessária para o método de Newton-Raphson
    """
    x = x_arg(m, T, g, j, lam, b)  # Calcula argumento
    dB_dx = dbril_dx(x, j)          # Derivada de Brillouin em relação a x
    dx_dm = (j * lam) / (g * Kb * T) # Derivada de x em relação a m
    return g * j * dB_dx * dx_dm     # Regra da cadeia: df/dm = df/dx * dx/dm

def g_of_m(m, T):
    """
    Função cujo ZERO queremos encontrar: g(m) = f(m) - m
    Solução autoconsistente satisfaz g(m) = 0
    """
    return f_of_m(m, T) - m

def gprime_of_m(m, T):
    """
    Derivada de g(m): g'(m) = f'(m) - 1
    """
    return fprime_of_m(m, T) - 1.0

# -----------------------------
# Método Newton-Raphson
# -----------------------------
def newton_raphson(m0, T, tol=1e-6, ite_max=5000):
    """
    Implementa o método de Newton-Raphson para encontrar raiz de g(m) = 0
    
    m0: chute inicial para magnetização
    T: temperatura atual
    tol: tolerância para convergência
    ite_max: número máximo de iterações
    
    Retorna: magnetização autoconsistente ou NaN se não convergir
    """
    m = float(m0)  # Garante que m é float
    
    # Loop de iterações do método Newton-Raphson
    for _ in range(ite_max):
        gm  = g_of_m(m, T)   # Calcula g(m)
        gpm = gprime_of_m(m, T)  # Calcula g'(m)
        
        # Prevenção contra divisão por zero ou derivada muito pequena
        if not np.isfinite(gpm) or abs(gpm) < 1e-12:
            # Se derivada problemática, usa média com método de ponto fixo
            m_next = 0.5*m + 0.5*f_of_m(m, T)
        else:
            # Fórmula padrão Newton-Raphson: m_{n+1} = m_n - g(m_n)/g'(m_n)
            m_next = m - gm/gpm

        # Critério de convergência: pequena variação E função próxima de zero
        if abs(m_next - m) < tol and abs(gm) < 1e-6:
            return m_next  # Convergiu!

        # Prevenção contra valores não-finitos (NaN, infinito)
        if not np.isfinite(m_next):
            m_next = f_of_m(m, T)  # Usa método de ponto fixo como fallback

        m = m_next  # Atualiza m para próxima iteração
    
    return np.nan  # Retorna NaN se não convergir após ite_max iterações

# -----------------------------
# Execução e comparação
# -----------------------------
def run_compare(m0=5.0, tol=1e-6):
    """
    Função principal que executa o cálculo para todas as temperaturas
    e plota os resultados
    
    m0: chute inicial para magnetização
    tol: tolerância para convergência
    """
    # Cria array vazio para armazenar resultados
    m_new = np.empty_like(T_grid, dtype=float)

    # Mede tempo inicial
    start = time.time()

    # Loop sobre todas as temperaturas na grade
    for i, Ti in enumerate(T_grid):
        # Para cada temperatura, calcula magnetização autoconsistente
        m_new[i] = newton_raphson(m0, Ti, tol=tol)

    # Mede tempo final e exibe
    end = time.time()
    print(f"Tempo de execução: {end - start:.4f} segundos")

    # Cria gráfico dos resultados
    plt.figure()
    plt.plot(T_grid, m_new, label='Newton-Raphson', linestyle='--')
    plt.xlabel('Temperatura (K)', fontsize=12)
    plt.ylabel('Magnetização', fontsize=12)
    plt.title('Magnetização × Temperatura', fontsize=14)
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.legend()    # Mostra legenda
    plt.tight_layout()  # Ajusta layout automaticamente
    plt.show()      # Exibe gráfico

    return m_new  # Retorna array com magnetizações calculadas

# Executa o código apenas se for o script principal
if __name__ == "__main__":
    # Chama função principal com chute inicial 5.0 e tolerância 1e-6
    m_new = run_compare(m0=5.0, tol=1e-6)