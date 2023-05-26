#４６ぬこだよ～ (っ◕‿◕)っ

import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) >= 2:
	steps = int(sys.argv[1])
	learning_rate = float(sys.argv[2])
else:
	steps = 5000
	learning_rate = 0.01

def f_true(x):
	return 2 + 0.8 * x

# conjunto de dados {(x,y)}
xs = np.linspace(-3, 3, 100)
ys = np.array([f_true(x) + np.random.randn() * 0.5 for x in xs])

# hipotese
def h(x, theta):
	return theta[0] + theta[1] * x

# funcao de custo
def J(theta):
	return (1/(2*m)) * summation_cost(theta)

m = len(ys)

def summation_cost(theta):
	return np.sum([(h(xs[i], theta) - ys[i]) ** 2 for i in range(m)])

def summation_theta_zero(theta):
	return np.sum([h(xs[i], theta) - ys[i] for i in range(m)])

def summation_theta_one(theta):
	return np.sum([(h(xs[i], theta) - ys[i]) * xs[i] for i in range(m)])

# derivada parcial com respeito a theta [i]
def gradient(theta):
	theta[0] -= learning_rate * (1/m) * summation_theta_zero(theta)
	theta[1] -= learning_rate * (1/m) * summation_theta_one(theta)
	
	return theta

""" plota no mesmo grafico : - o modelo / hipotese ( reta )
	- a reta original ( true function )
	- e os dados com ruido (xs , ys)
"""
def print_modelo(theta):
	plt.plot(xs, h(xs, theta), color='red', linewidth = '6')
	plt.plot(xs, h(xs, theta), color='green', linewidth = '3')
	plt.plot(f_true(xs[0]))
	plt.scatter(xs, ys)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Hypothesis (Red), Original Line (Green), and Data Points')
	plt.grid()
	plt.savefig("hypothesis_original_line_and_points.png")
	plt.show()

def plot_cost_function_over_steps():
	plt.plot(range(steps), cost_over_time)
	plt.xlabel('Steps')
	plt.ylabel('Cost')
	plt.title('Cost Function Value over Steps')
	plt.grid()
	plt.savefig("plot_cost_function_over_steps.png")
	plt.show()

theta = [np.random.randint(-3, 3), np.random.randint(-3, 3)]
theta_over_time = []
cost_over_time = []
hypothesys_over_time = []
theta_over_time.append(theta)

for step in range(steps):
	theta = gradient(theta)
	theta_over_time.append(theta)
	cost_over_time.append(J(theta))

print_modelo(theta)
plot_cost_function_over_steps()
