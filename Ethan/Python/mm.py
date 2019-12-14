import numpy as np 

size = input("Size: ")
N = int(size)
#Randomly generate matricies
matrix_a = np.random.rand(N, N)
matrix_b = np.random.rand(N, N)

matrix_a = matrix_a * 100
matrix_b = matrix_b * 100
matrix_a = matrix_a - 50
matrix_b = matrix_b - 50

print("Matrix_A:")
print(matrix_a)
print(" ")
print("Matrix_B")
print(matrix_b)
print(" ")
print("Result:")
print(matrix_a * matrix_b)
