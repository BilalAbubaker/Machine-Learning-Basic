# -*- coding: utf-8 -*-
"""
Created on Sun Jan 7 7:77:77 2024

@author: Shadi M. Alkhatib
"""

#                Pandas 
"""
* Pandas derived from Panel Data

* Pandas is an open-source Python Library providing tools for data analysis
  and manipulate with data using its data structure, Start developing in 2008
  
* These data structures are built on top of NumPy array, which means they 
  are fast.

*  Using Pandas, we can accomplish five typical steps in the processing and 
   analysis of data: load, prepare, manipulate, model, and analyze.

* The Power of Pandas:
1- Loading data from different file formats into memory in Data-Frame Object
2- Merging data with high performance
3- Handling missing data 
4- Insert, Update, and Delete columns and rows in Data-Frame
5- Group by data for aggregation and transformations
6- Export data into different files formats csv, xlsx,etc. 

* Pandas deals with three data structures :

1- Series : One-dimensional (1D) array with homogeneous data, size Immutable,
            values of data mutable

2- Data-Frame :  Two-dimensional array (2D) with heterogeneous data,
                 size mutable, data mutable

3- Panel : Three-dimensional data structure (3D) with heterogeneous data,
           size mutable, data mutable

* Data-Frame is widely used 
* Panel is used less
"""


# 1- Pandas – Series:
"""
1- Series is a one-dimensional labeled array 

2- Holding any type of data (integer, string, float, python objects, etc.)

3- A pandas Series can be created from ndarray, list, dictionary, constant

4- Accessing data in Series by indexing - same as ndarray
"""

# 1- Create Series
import pandas as pd

# Create empty series
S1 = pd.Series()

print(S1)   # Series([], dtype: object)


# 1.1 Create Series from dictionary
S1 = {'a':10, 'b':20, 'c': 30}

print(S1)                       #{'a': 10, 'b': 20, 'c': 30}

print(pd.Series( S1 ))
'''
a    10
b    20
c    30
dtype: int64
'''




# 1.2 Create Series from ndarray
import pandas as pd
import numpy as np


S2 = np.arange(5)

S3 = pd.Series(data = S2)

print(S3)
"""
0    0
1    1
2    2
3    3
4    4
dtype: int32
"""


S4 = np.array(['a', 'b', 'c'])

S5 = pd.Series(data = S4)

print(S5)
'''
0    a
1    b
2    c
dtype: object
'''

S6 = [0, 1, 2]

S7 = pd.Series( data = S4 , index = S6 )

print(S7)
'''
0    a
1    b
2    c
dtype: object
'''



S8 = {'a' : 0.0,  'b' : 1.0,  'c' : 2.}

S9 = pd.Series(S8.items(), index=[0,1,2])

print(S9)
"""
0    (a, 0.0)
1    (b, 1.0)
2    (c, 2.0)
dtype: object
"""



S10 = {'A' : 10.0, 'B' : 20.1, 'C' : 30}

S11 = pd.Series(S10.keys(),  index=[0, 1, 2])

print(S11)
"""
0    A
1    B
2    C
"""




# 1.3 Create Series from list
S12 = [34, 21, "40", 7.9]

S13 = pd.Series(data=S12)

print(S13)
"""
0     34
1     21
2     40
3    7.9
dtype: object
"""


# 1.4 Chane Name the Series
S14 = pd.Series(data = S12, name="Sham")

print(S14)
"""
0     34
1     21
2     40
3    7.9
Name: Sham, dtype: object
"""

print(S14.values)        # [34 21 '40' 7.9]

print(type(S14.values))  # <class 'numpy.ndarray'>





# 2- Indexing and Slicing Series
S15 = np.arange(5)

S16 = pd.Series(data=S15, index=['a', 'b','c', 'd', 'e'])

print(S16)
"""
a    0
b    1
c    2
d    3
e    4
dtype: int32
"""

print(S16[0:4])
"""
a    0
b    1
c    2
d    3
dtype: int32
"""


print(S16['a':'c'])
"""
a    0
b    1
c    2
dtype: int32
"""




for i in S16.index:
  print(i, " : " , S16[i])
"""
a  :  0
b  :  1
c  :  2
d  :  3
e  :  4
"""  


# 2.1 Boolean Indexing
print(S16 > 3)  
"""
a    False
b    False
c    False
d    False
e     True
dtype: bool
"""  


print(S16[ S16 > 3 ])
"""
ser[ ser > 15 ]
"""


# 2.2 Statistics in Pandas
print("Quantiles values : ")           # Quantiles values :

print("Q1 : ",  S16.quantile(0.25))   # Q1 :  1.0

print("Q2 : ",  S16.quantile(0.50))   # Q2 :  2.0

print("Q3 : ",  S16.quantile(0.75))  # Q3 :  3.0 



# 2.3 Summationtow series 

#Add two elements of series using list
S17 = pd.Series([1,2,3,4], index=['USA','Germany','USSR','Japan'])

print(S17)
'''
USA        1
Germany    2
USSR       3
Japan      4
dtype: int64
'''


S18 = pd.Series([1,2,5,4], index=['USA','Germany','Italy','Japan'])

print(S18)
'''
USA        1
Germany    2
Italy      5
Japan      4
dtype: int64
'''


#When two values differ, the NaN is marked as
# the value is converted from int to float
print(S17 + S18)
'''
Germany    4.0
Italy      NaN    # Not A Number
Japan      8.0
USA        2.0
USSR       NaN    # Not A Number
dtype: float64
'''








# 2- Pandas – Data Frame
"""
1- Data-Frame is a two-dimensional data structure

2-  Features of Data-Frame:
2.1- Potentially columns are of different types
2.2- Size – Mutable
2.3- Labeled axes (rows and columns)
2.4- Can Perform Arithmetic operations on rows and columns
	
3- Create Data-Frame using Lists, Dictionary, Series, ndarrays , Data-Frames

4- Data-Frames provide indexing, slicing , insert, update, delete columns

5- Pandas can create Data-Frames from csv, xlsx, JSON , database
"""

# 1- Create Data-Frame
import pandas as pd


# Create empty Data-Frame
df1 = pd.DataFrame()

print(df1)
"""
Empty DataFrame
Columns: []
Index: []
"""



# 1.1- Create Data-Frame from list

# Crate Data-Frame from 2D lists
S1 = [ ['Sham',98.0], ['Shadi',90.9], ['Rama',94.5] ]

df2 = pd.DataFrame( data = S1, columns=["name", "gpa"])

print(df2)
"""
    name    gpa 
    
0   Sham   98.0
1   Shadi  90.9
2   Rama   94.5
"""



# 2.1- Create Data-Frame from dictionary
S2 = { 'Name': ['Aws', 'Saif', 'Rakan', 'Rami'],
         'Age' : [9,27,29,40] 
       }


df3 = pd.DataFrame(S2)

print(df3)
"""
0    Aws    9
1   Saif   27
2  Rakan   29
3   Rami   40
"""


# Create a DataFrame from List of Dicts
S3 = [{'a': 5,  'b': 10, 'c': 15},
      {'a': 20, 'b': 25, 'c': 30},
      {'a': 35, 'b': 40} 
     ]

df4 = pd.DataFrame(S3)

print(df4)
"""
    a   b     c
0   5  10  15.0
1  20  25  30.0
2  35  40   NaN
"""  




# Create Data-Frame from Series with index
S4 = { 'one' : pd.Series( [1, 2] , ['a', 'b']),
      'two' : pd.Series( [1, 2, 3] , index=['a', 'b', 'c'])}

df5 = pd.DataFrame(data = S4)

print(df5)
"""
   one  two
a  1.0    1
b  2.0    2
c  NaN    3
"""


# Add column by Series
df5["three"] = pd.Series( [10,20,30], index=['a','b','c'])

print(df5)
"""
   one  two  three
a  1.0    1     10
b  2.0    2     20
c  NaN    3     30
"""

# Add column by constant
df5["four"] = 9

print(df5)
"""
   one  two  three  four
a  1.0    1     10     9
b  2.0    2     20     9
c  NaN    3     30     9
"""


# Delete Column from Data-Frame

# using del function
del df5["four"]

print(df5)
"""
   one  two  three
a  1.0    1     10
b  2.0    2     20
c  NaN    3     30
"""




# **Note Data frame is a group of series 

# Example using numpy and pandas 
import numpy  as np

import pandas as pd

from numpy.random import randn


df6 = pd.DataFrame( randn(5,4), 
                   index='A B C D E'.split(), 
                   columns='W X Y Z'.split() )


print(df6)
"""
          W         X         Y         Z
A  0.119145  0.665715  0.407586  0.807843
B -2.736464 -0.985153 -0.155276 -1.013228
C -1.087436 -2.567767  0.661029 -0.332921
D -0.928602  1.715950 -0.468756  0.860097
E  0.230087  0.618658 -2.052563 -0.166646
"""


print(df6['W'])
'''
A    0.119145
B   -2.736464
C   -1.087436
D   -0.928602
E    0.230087
Name: W, dtype: float64
'''


print(type(df6['W'] ))
# <class 'pandas.core.series.Series'>


print(type(df6))
# <class 'pandas.core.frame.DataFrame'>


# Series together
df6['New']=df6['W']+df6['Y']

print(df6['New'])
'''
A    0.526731
B   -2.891740
C   -0.426407
D   -1.397359
E   -1.822476
Name: New, dtype: float64
'''


print(df6)
'''
          W         X         Y         Z       New
A  0.119145  0.665715  0.407586  0.807843  0.526731
B -2.736464 -0.985153 -0.155276 -1.013228 -2.891740
C -1.087436 -2.567767  0.661029 -0.332921 -0.426407
D -0.928602  1.715950 -0.468756  0.860097 -1.397359
E  0.230087  0.618658 -2.052563 -0.166646 -1.822476
'''


# Drop column of dataframe
print(df6.drop('New', axis=1, inplace=True ))
# None

print(df6)
'''
          W         X         Y         Z
A  0.119145  0.665715  0.407586  0.807843
B -2.736464 -0.985153 -0.155276 -1.013228
C -1.087436 -2.567767  0.661029 -0.332921
D -0.928602  1.715950 -0.468756  0.860097
E  0.230087  0.618658 -2.052563 -0.166646
'''


# Drop Row of dataframe
print(df6.drop('E', axis=0, inplace=True ))
# None


print(df6)
'''
          W         X         Y         Z
A  0.119145  0.665715  0.407586  0.807843
B -2.736464 -0.985153 -0.155276 -1.013228
C -1.087436 -2.567767  0.661029 -0.332921
D -0.928602  1.715950 -0.468756  0.860097
'''


#Print  location of element
print(df6.loc['C'])
'''
W   -1.087436
X   -2.567767
Y    0.661029
Z   -0.332921
Name: C, dtype: float64
'''



#Print index  location of element
print(df6.iloc[2])
'''
W   -1.087436
X   -2.567767
Y    0.661029
Z   -0.332921
Name: C, dtype: float64
'''


#Print location of element in each index of element
print(df6.loc[['A','C'],['W','Y']])
'''
          W         Y
A  0.119145  0.407586
C -1.087436  0.661029
'''







# Task: How to work with Multi-index and index hierarchy
import pandas as pd

import numpy  as np


outside = ['G1','G1','G1','G2','G2','G2']

inside =[1,2,3,1,2,3]

hier_index=list(zip(outside,inside))

hier_index=pd.MultiIndex.from_tuples(hier_index)


print(outside)
# ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']



print(inside)
# [1, 2, 3, 1, 2, 3]




print(hier_index)
'''
MultiIndex([('G1', 1),
            ('G1', 2),
            ('G1', 3),
            ('G2', 1),
            ('G2', 2),
            ('G2', 3)],
           )
'''


print(list(zip(outside,inside)))
'''
[('G1', 1), ('G1', 2), ('G1', 3), ('G2', 1), ('G2', 2), ('G2', 3)]
'''



hier_index=pd.MultiIndex.from_tuples(hier_index)
print(hier_index)
'''
MultiIndex([('G1', 1),
            ('G1', 2),
            ('G1', 3),
            ('G2', 1),
            ('G2', 2),
            ('G2', 3)],
           )
'''


#The following example shows how to print random numbers inside index:
    
from numpy.random import randn

np.random.seed(101)


df = pd.DataFrame(np.random.randn(6,2),
                  index=hier_index,
                  columns=['A','B'])

print(df)
'''
             A         B
G1 1  2.706850  0.628133
   2  0.907969  0.503826
   3  0.651118 -0.319318
G2 1 -0.848077  0.605965
   2 -2.018168  0.740122
   3  0.528813 -0.589001
'''
   

print(df.loc['G1'])
'''
          A         B
1  2.706850  0.628133
2  0.907969  0.503826
3  0.651118 -0.319318
'''



print(df.loc['G1'].loc[1])
'''
A    2.706850
B    0.628133
Name: 1, dtype: float64
'''


df.index.names=['Group','Num']
print(df)
'''
                  A         B
Group Num                    
G1    1    2.706850  0.628133
      2    0.907969  0.503826
      3    0.651118 -0.319318
G2    1   -0.848077  0.605965
      2   -2.018168  0.740122
      3    0.528813 -0.589001
'''







# Create Data-Frame from csv file
df = pd.read_csv("california_housing_train.csv")


print(df)
"""
	longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
0	-114.31	34.19	15.0	5612.0	1283.0	1015.0	472.0	1.4936	66900.0
1	-114.47	34.40	19.0	7650.0	1901.0	1129.0	463.0	1.8200	80100.0
2	-114.56	33.69	17.0	720.0	174.0	333.0	117.0	1.6509	85700.0
3	-114.57	33.64	14.0	1501.0	337.0	515.0	226.0	3.1917	73400.0
4	-114.57	33.57	20.0	1454.0	326.0	624.0	262.0	1.9250	

[17000 rows x 9 columns]
"""


# head(): Return first n of rows from Data-Frame
#         By default return first 5 rows

print(df.head())
"""
longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
0	-114.31	34.19	15.0	5612.0	1283.0	1015.0	472.0	1.4936	66900.0
1	-114.47	34.40	19.0	7650.0	1901.0	1129.0	463.0	1.8200	80100.0
2	-114.56	33.69	17.0	720.0	174.0	333.0	117.0	1.6509	85700.0
3	-114.57	33.64	14.0	1501.0	337.0	515.0	226.0	3.1917	73400.0
4	-114.57	33.57	20.0	1454.0	326.0	624.0	262.0	1.9250	
"""



# Return first 3 rows
df.head(3)
"""
 longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
0	-114.31	34.19	15.0	5612.0	1283.0	1015.0	472.0	1.4936	66900.0
1	-114.47	34.40	19.0	7650.0	1901.0	1129.0	463.0	1.8200	80100.0
2	-114.56	33.69	17.0	720.0	174.0	333.0	117.0	1.6509	
"""



# tail(): Return last n of rows from Data-Frame
#         By default return first 5 rows

print(df.tail())
'''
longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
16995	-124.26	40.58	52.0	2217.0	394.0	907.0	369.0	2.3571	111400.0
16996	-124.27	40.69	36.0	2349.0	528.0	1194.0	465.0	2.5179	79000.0
16997	-124.30	41.84	17.0	2677.0	531.0	1244.0	456.0	3.0313	103600.0
16998	-124.30	41.80	19.0	2672.0	552.0	1298.0	478.0	1.9797	85800.0
16999	-124.35	40.54	52.0	1820.0	300.0	806.0	270.0	3.0147	
'''



# Return the last row in Data-Frame
print(df.tail(1))
'''
longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value
16999	-124.35	40.54	52.0	1820.0	300.0	806.0	270.0	3.0147	
'''


# Show all information the .csv file
print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17000 entries, 0 to 16999
Data columns (total 9 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   longitude           17000 non-null  float64
 1   latitude            17000 non-null  float64
 2   housing_median_age  17000 non-null  float64
 3   total_rooms         17000 non-null  float64
 4   total_bedrooms      17000 non-null  float64
 5   population          17000 non-null  float64
 6   households          17000 non-null  float64
 7   median_income       17000 non-null  float64
 8   median_house_value  17000 non-null  float64
dtypes: float64(9)
memory usage: 1.2 MB
"""



# Get shape of Data-Frame
print(df.shape)   #  (17000, 9)



# Get columns names from Data-Frame
list(df.columns)



# Select from one column
df.loc[ : , 'longitude']
'''
0       -114.31
1       -114.47
2       -114.56
3       -114.57
4       -114.57
          ...  
16995   -124.26
16996   -124.27
16997   -124.30
16998   -124.30
16999   -124.35
Name: longitude, Length: 17000, dtype: float64
'''



# Retrive sub data from spcific column
print(df.loc[5:15 , 'longitude'])
'''
5    -114.58
6    -114.58
7    -114.59
8    -114.59
9    -114.60
10   -114.60
11   -114.60
12   -114.61
13   -114.61
14   -114.63
15   -114.65
Name: longitude, dtype: float64
'''



# Select from multiple columns
print(df.loc[ :10 , ['total_rooms','total_bedrooms', 'population']])
'''
	total_rooms	total_bedrooms	population
0	5612.0	1283.0	1015.0
1	7650.0	1901.0	1129.0
2	720.0	174.0	333.0
3	1501.0	337.0	515.0
4	1454.0	326.0	624.0
5	1387.0	236.0	671.0
6	2907.0	680.0	1841.0
7	812.0	168.0	375.0
8	4789.0	1175.0	3134.0
9	1497.0	309.0	787.0
10	3741.0	801.0	2434.0
'''



# Describe all numaric data in Data-Frame
print(df.describe())
'''
           longitude	latitude	housing_median_age	total_rooms	total_bedrooms	population	households	median_income	median_house_value

count     17000.000000	17000.000000	17000.000000	17000.000000	17000.000000	17000.000000	17000.000000	17000.000000	17000.000000

mean      119.562108	35.625225	28.589353	2643.664412	539.410824	1429.573941	501.221941	3.883578	207300.912353

std	      2.005166	2.137340	12.586937	2179.947071	421.499452	1147.852959	384.520841	1.908157	115983.764387

min	     -124.350000	32.540000	1.000000	2.000000	1.000000	3.000000	1.000000	0.499900	14999.000000

25%	     -121.790000	33.930000	18.000000	1462.000000	297.000000	790.000000	282.000000	2.566375	119400.000000

50%	     -118.490000	34.250000	29.000000	2127.000000	434.000000	1167.000000	409.000000	3.544600	180400.000000

75%	     -118.000000	37.720000	37.000000	3151.250000	648.250000	1721.000000	605.250000	4.767000	265000.000000

max	     -114.310000	41.950000	52.000000	37937.000000	6445.000000	35682.000000	6082.000000	15.000100	500001.000000
'''










# Manipulate with missing data
import pandas as pd

import numpy as np


# Check null values in Data-Frame
data = {'one'   : pd.Series([1, 2, 3],    index=['a', 'b', 'c']),
        'two'   : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']),
        'three' :  pd.Series([10,20,30],  index=['a','b','c']) }

df_temp = pd.DataFrame(data)

print(df_temp)
'''
   one  two  three
a  1.0    1   10.0
b  2.0    2   20.0
c  3.0    3   30.0
d  NaN    4    NaN
'''


print(type(df_temp["three"]['d']))
# <class 'numpy.float64'>


print(df_temp.isnull())
'''
     one    two  three
a  False  False  False
b  False  False  False
c  False  False  False
d   True  False   True
'''


print(df_temp.notnull())
'''
     one   two  three
a   True  True   True
b   True  True   True
c   True  True   True
d  False  True  False
'''



# Fill NaN values with 0s.
print(df_temp.fillna(value = 0))
'''
   one  two  three
a  1.0    1   10.0
b  2.0    2   20.0
c  3.0    3   30.0
d  0.0    4    0.0
'''


print(df_temp["three"].mean())
# 20.0


# Fill all NaN values with with mean value of specific column
print(df_temp.fillna(value = df_temp["three"].mean() ))
'''
    one  two  three
a   1.0    1   10.0
b   2.0    2   20.0
c   3.0    3   30.0
d  20.0    4   20.0
'''








# Graphics
import pandas as pd

data = pd.Series((3,6,9,8,5,4,2,6,3,5,8))

data.plot()  

data.plot(kind='line')


#    OR
import pandas as pd

data = pd.Series((3,6,9,8,5,4,2,6,3,5,8))

data.plot(kind='pie') 

#   OR Try
'''
data.plot(kind='bar')
data.plot(kind='barh')
data.plot(kind='hist')
data.plot(kind='kde')
data.plot(kind='area')
'''
