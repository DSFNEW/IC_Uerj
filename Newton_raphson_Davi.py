# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:15:26 2025
@author: Davi
"""

import matplotlib.pyplot as plt  # Biblioteca para plotagem de gráficos
import numpy as np  # Biblioteca para cálculos numéricos
import timeit  # Biblioteca para medir tempo de execução

# Constantes físicas utilizadas nos cálculos
MIb = 0.0578838  # Momento magnético de Bohr (em eV/T)
Kb = 0.086       # Constante de Boltzmann (em eV/K)
tol = 1e-6       # Tolerância para convergência do método numérico
m0 = 0.000001    # Valor inicial de magnetização (menor que no código anterior)

def funcX(g, j, T, lam, b, m):
    Bef = m * (lam / (g**2 * MIb)) + b
    return (g * MIb * j * Bef) / Kb * T

# Função de Brillouin: descreve o comportamento magnético de sistemas quânticos
def func_Bril(g, j, T, lam, b, m):
    A = (2 * j + 1) / (2 * j)  # Coeficiente A da função de Brillouin
    B = 1 / (2 * j)            # Coeficiente B da função de Brillouin
    x = funcX(g, j, T, lam, b, m)  # Argumento adimensional
    # Cálculo da função de Brillouin
    Bri = (1 / (np.tanh(A * x))) * A - (1 / np.tanh(B * x)) * B
    return Bri

def f_bri_deriv(g, j, T, lam, b, m):
    
    A = (2 * j + 1) / (2 * j)  
    B = 1 / (2 * j)            
    x = funcX(g, j, T, lam, b, m)
    
    return -(A**2)/np.sinh(A*x)**2 + (B**2)/np.sinh(B*x)**2
    
# Função que calcula a magnetização usando a função de Brillouin
def func_mag_iterac(g, j, T, lam, b, m):    
        
    return g * j * func_Bril(g, j, T, lam, b, m) 

# Função que calcula a derivada da magnetização (para o método de Newton-Raphson)
def func_mag_deriv(g, j, T, lam, b, m):
    A = (2 * j + 1) / (2 * j)  # Coeficiente A
    B = 1 / (2 * j)            # Coeficiente B
    x = funcX(g, j, T, lam, b, m)
    # Derivada da função de Brillouin em relação à magnetização
    Mag_dev = (j**2 * lam / Kb * T) * ((-1 / ((np.sinh(A * x)) ** 2)) * A **2) + ((1 / ((np.sinh(B * x)) ** 2)) * B ** 2)
    return Mag_dev
    
# Função de iteração com parâmetros pré-definidos para o sistema
def funcao_iteracao(m0, T):
    g = 2      # Fator g de Landé
    j = 7/2    # Momento angular total (para íons de Gadolínio)
    b = 0      # Campo magnético externo (zero neste caso)
    lam = 2.77 # Parâmetro de campo molecular
    return func_mag_iterac(g, j, T, lam, b, m0)    

# Função derivada com parâmetros pré-definidos
def func_iteracao_deriv(m0, T):
    g = 2
    j = 7/2
    b = 0
    lam = 2.77   
    return func_mag_deriv(g, j, T, lam, b, m0)
    
# Implementação do método de Newton-Raphson para encontrar a raiz
def func_newton_raphson(m0, T, tol):
    iter_count = 0  # Contador de iterações
    
    while True:
        # Fórmula de Newton-Raphson: m_{n+1} = m_n - f(m_n)/f'(m_n)
        m = m0 - (funcao_iteracao(m0, T) / func_iteracao_deriv(m0, T))
        iter_count += 1

        # Critério de convergência: diferença menor que a tolerância ou máximo de iterações
        if abs(m - m0) < tol or iter_count >= 5000:
            break

        m0 = m  # Atualizar o valor para a próxima iteração
    
    # Verificar se convergiu
    if iter_count >= 5000:
        print("O método não convergiu após 5000 iterações.")
        return None
    else:
        return m  # Retornar o valor convergido

# Função principal
def main():
    # Faixa de temperaturas de 1K até 200K com incrementos de 1K
    T = np.arange(1., 200.1, 1.)

    # Array para armazenar os valores de magnetização
    save = [0.0] * len(T)

    # Calcular a magnetização para cada temperatura
    for i in range(len(T)):
        itr = func_newton_raphson(m0, T[i], tol)
        save[i] = itr

    # Configurar e mostrar o gráfico
    plt.xlabel('Magnetização', fontsize=12)
    plt.ylabel('Temperatura (Kelvin)', fontsize=12)
    plt.title('Relação entre Magnetização e Temperatura', fontsize=14)
    plt.plot(T, save)  # Plotar magnetização versus temperatura
    plt.show()
    
# Medir o tempo de execução
time = timeit.timeit(main, number=1)
print(f"O tempo de execução foi de {time:.2f} segundos")