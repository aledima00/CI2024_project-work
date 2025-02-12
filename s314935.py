import numpy as np

def f1(x:np.ndarray)->np.ndarray:
	# estimated function for problem1
	# mse: 0.000e+00
	return np.sin(x[0])

def f2(x:np.ndarray)->np.ndarray:
	# estimated function for problem2
	# mse: 2.962e+13
	return x[0]

def f3(x:np.ndarray)->np.ndarray:
	# estimated function for problem3
	# mse: 5.166e+02
	return np.multiply(x[1],np.subtract(x[1],np.multiply(x[1],x[1])))

def f4(x:np.ndarray)->np.ndarray:
	# estimated function for problem4
	# mse: 2.162e+01
	return 2.0

def f5(x:np.ndarray)->np.ndarray:
	# estimated function for problem5
	# mse: 5.573e-18
	return 0.0

def f6(x:np.ndarray)->np.ndarray:
	# estimated function for problem6
	# mse: 1.339e+00
	return np.add(np.minimum(np.subtract(2.0,x[0]),x[1]),np.minimum(x[1],x[1]))

def f7(x:np.ndarray)->np.ndarray:
	# estimated function for problem7
	# mse: 7.127e+02
	return 9.0

def f8(x:np.ndarray)->np.ndarray:
	# estimated function for problem8
	# mse: 2.299e+07
	return x[5]

