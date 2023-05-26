# (っ◕‿◕)っ

import numpy as np
import sys

if len(sys.argv) >= 2:
	steps = int(sys.argv[1])
	learning_rate = int(sys.argv[2])
else:
	steps = 5000
	learning_rate = 0.01

def f_true(x):
	return 2 + 0.8 * x

# conjunto de dados {(x,y)}
xs = np.linspace(-3, 3, 100)
ys = np.array([f_true(x) + np.random.randn() * 0.5 for x in xs])
m = len(ys)
theta_over_time = []
cost_over_time = []
hypothesys_over_time = []

# hipotese
def h(x, theta):
	return theta[0] + theta[1] * x

# funcao de custo
def J(theta):
	return (1/(2*m)) * summation_cost(theta)

def summation_cost(theta):
	return np.sum([(h(xs[i], theta) - ys[i]) ** 2 for i in range(m)])

def summation_theta_zero(theta):
	return np.sum([h(xs[i], theta) - ys[i] for i in range(m)])

def summation_theta_one(theta):
	return np.sum([(h(xs[i], theta) - ys[i]) * xs[i] for i in range(m)])

# derivada parcial com respeito a theta [i]
def gradient(i, theta):
	theta[0] -= learning_rate * (1/m) * summation_theta_zero(theta)
	theta[1] -= learning_rate * (1/m) * summation_theta_one(theta)
	
	return theta

""" plota no mesmo grafico : - o modelo / hipotese ( reta )
	- a reta original ( true function )
	- e os dados com ruido (xs , ys)
"""
def print_modelo(theta):
	pass

theta = [np.random.randint(-3, 3), np.random.randint(-3, 3)]
theta_over_time.append(theta)

for step in range(steps):
	theta = gradient(step, theta)
	theta_over_time.append(theta)
	hypothesys_over_time.append(h(theta))
	cost_over_time.append(J(theta))