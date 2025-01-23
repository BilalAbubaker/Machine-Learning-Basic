# -*- coding: utf-8 -*-
"""
Created on Sat Jan 6 6:66:66 2024

@author: Shadi M. Alkhatib
"""

#                NumPy Arrays
"""
* NumPy : Numerical Python.

* NumPy : provides a high-performance multidimensional array.

* NumPy : aims to provide an array object that is up to 50x faster
          than traditional Python lists.
          
* NumPy : provides fast and efficient operations on arrays of 
          homogeneous data, which means data inside an array 
          must be of the same data type.

* Why is NumPy Faster Than Lists?
1- NumPy arrays are stored at one continuous place in memory
   unlike lists.
   
2- NumPy is a Python library and is written partially in Python,
   but most of the parts that require fast computation are 
   written in C or C++.
   

* NumPy arrays and lists are data structures allow indexing,
      slicing, and iterating.
"""

# Create 1D array
# pip install numpy    
import numpy as np  


S1 = np.array([1,2,3,4,5])

print(S1)        # [1 2 3 4 5]

print(type(S1))  # <class 'numpy.ndarray'>

print(S1.dtype)  # int32

print(S1.ndim)   # 1

print(S1.shape)  # (5,)

print(len(S1))   # 5



S2 = np.array([1,2,3,4,5.5])

print(S2)        # [1.  2.  3.  4.  5.5]

print(S2.dtype)  # float64




S3 = np.array([1,2,3,"4",5.0])

print(S3)        # ['1' '2' '3' '4' '5.0']

print(S3.dtype)  # <U32



S4 = np.array([1,2,3,"4",5.0], dtype=np.int16)

print(S4)   # [1 2 3 4 5]



#S5 = np.array([1,2,3,"4","5.0"], dtype=np.int16)
#print(S5)  # invalid literal for int() with base 10: '5.0'



S6 = np.array([1,2,3,"4","5.0"], dtype=np.float16)

print(S6)




print(np.array([1, 2, 3]))       #[1 2 3]

print(np.arange(10))  # [0 1 2 3 4 5 6 7 8 9]

print(np.arange(15).reshape(3, 5)) 
"""
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
"""

print(np.arange(100).reshape(10,10))  
"""
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
"""

print(np.random.randint(1,100,6))     # [79 73 19 17 61 82] 

print(np.random.randint(1,100,6).reshape(2,3)) 
"""
[[47 47  9]
 [62 76  2]]
"""

print(np.random.rand(3)) # [0.24278335 0.52675106 0.08927907]

print(np.ones(3))     # [1. 1. 1.]

print(np.ones((2,2)))     
"""
[[1. 1.]
 [1. 1.]]
"""

print(np.zeros(3))   # [0. 0. 0.]

print(np.zeros((3,3)))                         
"""
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]] 
"""

print(np.zeros(shape=(3,2,3)))
"""
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]]
"""

print(np.arange(0, 2, 0.3))  # [0.  0.3 0.6 0.9 1.2 1.5 1.8]





#Basic Operations of ndarray:
A = np.array([[1,1],[0,1]]) 
 
B = np.array([[2,0],[3,4]])


print(np.add(A,B)) # Try  print(np.subtract(A,B)) / np.multiply/ np.divide
"""
[[3 1]
 [3 5]]
"""

print(A + B) # Try      print(A - B)  print(A * B)  print(A / B) 
"""
[[3 1]
 [3 5]]
"""

print(B < 3) 
"""
[[ True  True]
 [False False]]
""" 

print(A.sum() )    # 3

print(A.min() )    # 0  

print(A.argmin() ) # 2 

print(A.max() )    # 1  

print(A.argmax() )  # 0

print(np.average(A)) # 0.75

print(np.sqrt(B))  
"""
[[1.41421356   0.  ]
 [1.73205081   2.  ]]
"""





# Indexing and Slicing using numpy:
S7 = np.arange( 4 ) ** 2  


print(S7)          # [0 1 4 9]
       
print(S7[2])       # 4

print(S7[::-1] )   # [9 4 1 0]

print(S7[S7>5] )   # [9]  


S8 = S7[:]  
print(S8)         # [0 1 4 9]


S8[:] = 99  
print(S8)        # [99 99 99 99]


print(S7)       # [99 99 99 99]





# Create 2D ndarray with data type
S9 = np.array( [ [0,1,2,3,4], [5,6,7,8,9] ] )


print(S9)
"""
[[0 1 2 3 4]
 [5 6 7 8 9]]
"""

print(S9.dtype)   # int32


# Convert data from Integer into String
S9 = np.array([[0,1,2,3,4,], [5,6,7,8,255]], dtype = 'U') # Unicode String


print(S9)
"""
[['0' '1' '2' '3' '4']
 ['5' '6' '7' '8' '255']]
"""

print(S9.dtype)   # <U3


# Convert data from string into int16
S10 = np.array(["10", "20", "30", "40", "50"], dtype=np.int16)


print(S10)         # [10 20 30 40 50]

print(S10.dtype)  # int16



# Convert data from string into float16
S10 = np.array(["10", "20", "30", "40", "50"], dtype='f2')


print(S10)         # [10. 20. 30. 40. 50.]

print(S10.dtype)   # float16



# Basic Operations on a 2D Array:
S11 = np.array([[3,2],[0,1]])  

S12 = np.array([[3,1],[2,1]])  

  
print(S11)  
"""
[[3 2]
 [0 1]] 
"""

print(S12)  
"""
[[3 1]
 [2 1]] 
"""

print (S11 + S12)   # Try  print (A*B)  
"""
[[6 3]
 [2 2]] 
"""

S11 +=2  
print(S11,'\n')  
"""
[[5 4]
 [2 3]] 
"""





# NumPy - Array Attributes
S13 = np.array([1,2,3,4])


print(S13)          # [1 2 3 4]

print(S13.shape)    # (4,)


S13 = np.array([[1,2,3,4], [5,6,7,8]])


print(S13)
"""
[[1 2 3 4]
 [5 6 7 8]]
"""

print(S13.shape)    # (2, 4)

# This array attribute returns a tuple consisting of S13ay dimensions
print(S13.shape)   # (2, 4)

# ndarray.shape can be used to reshape the S13ay.
S13.shape = (4,2)

print(S13)
"""
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
"""

print(S13.shape)   # (4, 2)



# numpy.arang(srart, end, steps)
S14 = np.arange(0,10)


print(S14)         # [0 1 2 3 4 5 6 7 8 9]

print(S14.shape)  # (10,)

S14.shape = (2,5)
print(S14)
"""
[[0 1 2 3 4]
 [5 6 7 8 9]]
"""

S14.shape = (10,)

print(S14)   # [0 1 2 3 4 5 6 7 8 9]



S15 = np.arange(0,24)


print(S15)  
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]


print(S15.shape)   # (24,)



S15.shape = (8,3) #(4,6) # (2,12)

print(S15)
""" 
 [[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [12 13 14]
 [15 16 17]
 [18 19 20]
 [21 22 23]]
"""


S15.shape = (2,4,3)

print(S15.size)    # 24

print(S15)
"""
 [[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]
  [ 9 10 11]]

 [[12 13 14]
  [15 16 17]
  [18 19 20]
  [21 22 23]]]
"""





# Numpy Array Indexing and Slicing
S16 = np.arange(25)


print(S16) 
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]


print(S16[10])
# 10

print(S16[2:11])  # [ 2  3  4  5  6  7  8  9 10]

print(S16[2:11:2])  # [ 2  4  6  8 10]






S17 = S16.reshape(5,5)


print(S17)
"""
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
"""

print(S17[1])      # [5 6 7 8 9]

print(S17[1][2])  # 7

print(S17[1, 2]) # 7

print(S17[0][1])  # 1

print(S17[0 , 1]) # 1

print("number of dimensions : " ,S17.ndim) # number of dimensions :  2

print("shape of the array : ", S17.shape) # shape of the array :  (5, 5)

print("number of dimensions : " ,len(S17.shape)) # number of dimensions :  2

print(S17[ 0 , 1:4])  # [1 2 3]

print(S17[0:2])
"""
[[0 1 2 3 4]
 [5 6 7 8 9]]
"""

print(S17[0:2, 1:4])
"""
[[1 2 3]
 [6 7 8]]
"""



# Create 3D array
"""
We have 3 dimensions
arr[ first dimension , second dimension , third dimension] |or| 
arr[first dimension][second dimension][third dimension]
for each dimension [start : end : steps]
"""

S18 = np.arange(27).reshape(3,3,3)

print(S18)
"""
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]
"""

print(S18[0])
"""
[[0 1 2]
 [3 4 5]
 [6 7 8]]
"""

print(S18[0,1])        # [3 4 5]

print(S18[0, 1, -1])  # 5      Last item

print(S18[1, 2, 1])  #16

print(S18)
"""
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]
"""

# Access to the 4 value in array
print(S18[0][1][1])    # 4
print(S18[0 , 1 , 1])  # 4


# How to access [[ 0,  1,  2],
#                [ 3,  4,  5]]
print(S18[0, 0:2])
"""
[[0 1 2]
 [3 4 5]]
"""

# How to access  [[15, 16]] list
print(S18[1, 2, 0:2])
# [15 16]


# How to access [[18, 19, 20],
#                [21, 22, 23]]

print(S18[-1, :2, :])
"""
[[18 19 20]
 [21 22 23]]
"""


# How to access to [18, 21]
print(S18[-1, 0:2, 0])         # [18 21]


# How to access first two columns in all rows
print(S18[:, :, :2])
"""
[[[ 0  1]
  [ 3  4]
  [ 6  7]]

 [[ 9 10]
  [12 13]
  [15 16]]

 [[18 19]
  [21 22]
  [24 25]]]
"""




S19 = np.arange(125)
S19.shape = (25,5)


print(S19)
"""
[[  0   1   2   3   4]
 [  5   6   7   8   9]
 [ 10  11  12  13  14]
 [ 15  16  17  18  19]
 [ 20  21  22  23  24]
 [ 25  26  27  28  29]
 [ 30  31  32  33  34]
 [ 35  36  37  38  39]
 [ 40  41  42  43  44]
 [ 45  46  47  48  49]
 [ 50  51  52  53  54]
 [ 55  56  57  58  59]
 [ 60  61  62  63  64]
 [ 65  66  67  68  69]
 [ 70  71  72  73  74]
 [ 75  76  77  78  79]
 [ 80  81  82  83  84]
 [ 85  86  87  88  89]
 [ 90  91  92  93  94]
 [ 95  96  97  98  99]
 [100 101 102 103 104]
 [105 106 107 108 109]
 [110 111 112 113 114]
 [115 116 117 118 119]
 [120 121 122 123 124]]
"""

# How to access to all items in first column
print(S19[:, 0])
"""
[  0   5  10  15  20  25  30  35  40  45  50  55  60  
  65  70  75  80  85 90  95 100 105 110 115 120]
"""


# How to access to all items in first column with step 3
print(S19[::3, 0]) # [  0  15  30  45  60  75  90 105 120]


print(S19) # Show all data 

# Task
# write python to access to the following list : [ 4,14,24,34,44,54,64,74]

print(S19[0:16:2, -1])  # [ 4 14 24 34 44 54 64 74]






# NumPy – Array Manipulation

# 1- Resahpe arrays

# 1.1- ndarrray.reshape
S20 = np.arange(100)


print(S20)
"""
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
"""

print(S20.shape) # (100,)

print(S20.reshape(5,2,10))
"""
[[[ 0  1  2  3  4  5  6  7  8  9]
  [10 11 12 13 14 15 16 17 18 19]]

 [[20 21 22 23 24 25 26 27 28 29]
  [30 31 32 33 34 35 36 37 38 39]]

 [[40 41 42 43 44 45 46 47 48 49]
  [50 51 52 53 54 55 56 57 58 59]]

 [[60 61 62 63 64 65 66 67 68 69]
  [70 71 72 73 74 75 76 77 78 79]]

 [[80 81 82 83 84 85 86 87 88 89]
  [90 91 92 93 94 95 96 97 98 99]]]
"""

print(S20.shape)   # (100,)


S21 = S20.reshape(10, 10)

print(S21.shape)   # (10, 10)

print(S21)
"""
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
"""




# 1.2- ndarray.flatten
S22 = np.arange(100).reshape(10,10)


print(S22)
"""
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
"""


print(S22[0, :]) # [0 1 2 3 4 5 6 7 8 9]  # Flatten row [0]
 

list_flatten = []
for row in S22:
  list_flatten.extend(row)

print(np.array(list_flatten))
"""
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
"""


print(S22[:, 0]) #[ 0 10 20 30 40 50 60 70 80 90] # # Flatten column [1]


list_flatten = []
for col_idx in range(S22.shape[1]):
  list_flatten.extend(S22[:, col_idx])

print(np.array(list_flatten))
"""
[ 0 10 20 30 40 50 60 70 80 90  1 11 21 31 41 51 61 71 81 91  2 12 22 32
 42 52 62 72 82 92  3 13 23 33 43 53 63 73 83 93  4 14 24 34 44 54 64 74
 84 94  5 15 25 35 45 55 65 75 85 95  6 16 26 36 46 56 66 76 86 96  7 17
 27 37 47 57 67 77 87 97  8 18 28 38 48 58 68 78 88 98  9 19 29 39 49 59
 69 79 89 99]
"""


print(S22.flatten(order="C"))# C : C style (Default)[0] / F :Fortan style)[1]
"""
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
"""


print(S22.flatten(order="F"))
"""
[ 0 10 20 30 40 50 60 70 80 90  1 11 21 31 41 51 61 71 81 91  2 12 22 32
 42 52 62 72 82 92  3 13 23 33 43 53 63 73 83 93  4 14 24 34 44 54 64 74
 84 94  5 15 25 35 45 55 65 75 85 95  6 16 26 36 46 56 66 76 86 96  7 17
 27 37 47 57 67 77 87 97  8 18 28 38 48 58 68 78 88 98  9 19 29 39 49 59
 69 79 89 99]
"""



# 1.3- ndarray.T | ndarray.transpose
S23 = np.arange(10).reshape(5,2)


print(S23)
"""
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
"""


print(S23.transpose())   # Try  print(arr.T)
"""
[[0 2 4 6 8]
 [1 3 5 7 9]]
"""



# 2- Joining arrays

# 2.1 Concatenate
S24 = np.arange(6).reshape(2,3)
S25 = np.arange(6,12).reshape(2,3)

print(S24)
"""
[[0 1 2]
 [3 4 5]]
"""

print(S25)
"""
[[ 6  7  8]
 [ 9 10 11]]
"""

print(S24.shape)  # (2, 3)


# concate on axis 1
arr_conc_S26 = np.concatenate((S24, S25), axis=1)

print(arr_conc_S26)
"""
[[ 0  1  2  6  7  8]
 [ 3  4  5  9 10 11]]
"""

print(arr_conc_S26.shape)   # (2, 6)



# concate on axis 0
arr_conc_S26 = np.concatenate((S24, S25), axis=0)

print(arr_conc_S26)
"""
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
"""

print(arr_conc_S26.shape)   # (4, 3)




# 2.2 stack
S27 = np.arange(5)

S28 = np.arange(5,10)


print(S27)         # [0 1 2 3 4]

print(S27.shape)  # (5,)


print(S28)        # [5 6 7 8 9]

print(S28.shape)  #(5,)


stacked_arr_0 = np.stack([S27, S28], axis=0)


print(stacked_arr_0 )
"""
 [[0 1 2 3 4]
  [5 6 7 8 9]]
"""


print(S27)     # [0 1 2 3 4]

print(S28)     # [5 6 7 8 9]

stacked_arr_1 = np.stack([S27, S28], axis=1)

print(stacked_arr_1 )
"""
[[0 5]
 [1 6]
 [2 7]
 [3 8]
 [4 9]]
"""



# 2.3 horizontal stack
print(np.hstack([S27,S28])) # [0 1 2 3 4 5 6 7 8 9]



#2.4 vertical stack
print(np.vstack([S27,S28]))
"""
[[0 1 2 3 4]
 [5 6 7 8 9]]
"""







#3- Splitting arrays
S29 = np.arange(12)


print(S29)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

print(S29.shape)  # (12,)


# Split the array in 2 equal-sized subarrays
print(np.split(S29, 2, axis=0))
"""
[array([0, 1, 2, 3, 4, 5]), 
 array([ 6,  7,  8,  9, 10, 11])]
"""


# Split the S29ay in 3 equal-sized subarrays
print(np.split(S29, 3, axis=0))
"""
[array([0, 1, 2, 3]), 
 array([4, 5, 6, 7]), 
 array([ 8,  9, 10, 11])]
"""

# Split the array in 4 equal-sized subarrays
print(np.split(S29, 4, axis=0))
"""
[array([0, 1, 2]), 
 array([3, 4, 5]), 
 array([6, 7, 8]), 
 array([ 9, 10, 11])]
"""




# 4- Adding \ Removig Elements

# 4.1- resize
S30 = np.array([[1,2,3],[4,5,6]])

print(S30)
"""
[[1 2 3]
 [4 5 6]]
"""

print(S30.shape)  # (2, 3)

print(np.resize(S30,(3,4)))
"""
[[1 2 3 4]
 [5 6 1 2]
 [3 4 5 6]]
"""

print(np.resize(S30,(3,8)))
"""
[[1 2 3 4 5 6 1 2]
 [3 4 5 6 1 2 3 4]
 [5 6 1 2 3 4 5 6]]
"""



# 4.2- Append
S31 = np.array([[1,2,3],[4,5,6]])


S31 = np.append(S31, [7,8,9])

print(S31)   # [1 2 3 4 5 6 7 8 9]




# 4.3- Insert
S32 = np.array([[1,2,3],[4,5,6]])


print(S32)
"""
[[1 2 3]
 [4 5 6]]
"""

print(S32.flatten())  # [1 2 3 4 5 6]


print(np.insert(S32 , 2 , [1,1,1]))  # [1 2 1 1 1 3 4 5 6]



# 4.4- Delete
S33 = np.arange(12).reshape(3,4)


print(S33)
"""
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
"""

print(S33.flatten())  # [ 0  1  2  3  4  5  6  7  8  9 10 11]


# Array flattened before delete operation as axis not used
print(np.delete(S33,5)) #[ 0  1  2  3  4  6  7  8  9 10 11]





# 4.5- Unique
S34 = np.array([5,2,6,2,7,5,6,8,2,9,9,8,9])


print(S34) # [5 2 6 2 7 5 6 8 2 9 9 8 9]

print(np.unique(S34))  # [2 5 6 7 8 9]


'''
# NumPy I/O

# Save array as npy file
S33 = np.arange(12).reshape(3,4)


print(S33)
"""
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
"""

# Save array in a disk file with npy extension
S34= np.save("output_array", S33)



# Load array from npy file
S35 = np.load("output_array.npy")
'''














