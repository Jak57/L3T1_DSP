# importing numpy package
import numpy as np

# numpy array is a grid of values, all of the same type and is indexed by a tuple
# of nonnegative integers

# # no. of dimensions is the rank of the array
# a = np.array([1, 2, 3]) # create a rank 1 array
# print(a, type(a), a[0], a[1], a[2])
# a[0] = 5
# print(a)

# # the shape of an array is a tuple of integers giving the size of the array along 
# # each dimension
# print(a.shape)

# b = np.array([[1, 2, 3], [4, 5, 6]]) # create a rank 2 array
# print(b, b.shape)
# print(b[0, 0], b[0, 1], b[1, 0])


##########
# # numpy also provides many functions to create arrays
##########

# a = np.zeros((2, 2)) # create an array of all zeros
# b = np.zeros((4, 3, 2)) # have to specify shape between parentheses
# print(a)
# print(b)

# a = np.ones((3, 4, 3)) # create an array of all ones
# print(a)

# b = np.full((3, 5), 7) # create a constant array
# print(b)

# c = np.eye(5) # create a 5 X 5 
# print(c)

# e = np.random.random((2, 3))
# print(e)


########
# Array Indexing
########

# As arrays can be multi-dimensional, we have to specify slice
# for each dimension
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# use slicing to pull out the the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2)

# b = a[:2, 1:3]
# print(b)

# # a slice of an array is a view into the same date, so modifying it will modify
# # the original array
# print(a[0, 1]) 
# b[0, 0] = 77
# print(a[0, 1])


# you can also mix integer indexing with slicing, doing so will yield an array of
# lower rank than the original array

# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# print(a)

# row_1 = a[1, :] # rank 1 view of the second row of a
# print(row_1, type(row_1), row_1.shape)

# row_2 = a[1:2, :] # rank 2 view of the second row of a
# print(row_2, row_2.shape)

# row_3 = a[[1], :] # rank 2 view of the second row of a
# print(row_3, row_3.shape)


# # we can make same distinction when accessing columns of an array

# col_1 = a[:, 1] # rank 1 view of the second column of a
# print(col_1, col_1.shape)

# col_2 = a[:, 1:2] # rank 2 view of the scond column of a
# print(col_2, col_2.shape)

# col_3 = a[:, [1]] # rank 2 view of the second column of a
# print(col_3, col_3.shape)


#######
# Integer array indexing
#######

# # integer array indexing allows to create arbitrary array using the data from another
# # array

# a = np.array([[1, 2], [3, 4], [5, 6]])
# print(a, a.shape)

# print(a[[0, 1, 2], [0, 1, 0]]) # the return array will have shape (3, )

# # the above example of array indexing is similar to this:
# print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# # when using integer array indexing we can reuse the same element from the
# # original array

# print(a[[0, 0], [1, 1]])

# # another approach
# print(np.array([a[0, 1], a[0, 1]]))


# One useful trick of array indexing is selecting or mutating one element
# from each row of a matrix

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# print(a)

# b = [0, 2, 0, 1] # create an array of indices
# c = a[np.arange(4), b]
# print(c, c.shape, type(c)) # select one element from each row of a using indices in b
# print(np.arange(4))
# print("---------\n")

# # mutate one element from each row of a using the indices in b
# a[np.arange(4), b] += 10
# print(a, a.shape, type(a))


##########
# Boolean array indexing
##########

# boolean array indexing lets you pick arbitrary element of an array. Frequently
# this type of indexing is used to select the elements of an array that satisfy
# some condition

# a = np.array([[1, 2], [3, 4], [5, 6]])

# # Find the elements of a that are bigger than 2; this returns a numpy array of
# # boolean of the same shape as a, where each slot of bool_idx tells whether 
# # that element of a is greater than 2
# bool_idx = (a > 2)
# print(bool_idx)

# # we use boolean array indexing to construct a rank 1 array
# # constituting the elements of a corresponsing to the True value in bool_idx
# print(a[bool_idx])

# # another approach
# print(a[a > 2])
# print(a[a < 2])


##########
# Datatypes
##########

# a = np.array([1, 2]) # Let numpy choose the datatype
# b = np.array([1.0, 2.0]) # Let numpy choose the datatype
# c = np.array([1, 2], dtype=np.int32)
# print(a.dtype, b.dtype, c.dtype)


#########
# Array Math
#########

# x = np.array([[1, 2], [3, 4]], dtype=np.float64)
# y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# print(x)
# print(y)
# print()

# # Elementwise sum; both produce the same array
# print(x + y)
# print(np.add(x, y))
# print()

# # Elementwise difference, both produce the array
# print(x - y)
# print(np.subtract(x, y))
# print()

# # Elementwise multiplication, both produce the array
# print(x * y)
# print(np.multiply(x, y))
# print()

# # Elementwise division, both produces the array
# print(x / y)
# print(np.divide(x, y))
# print()

# # Elementwise square root, produces the array
# print(np.sqrt(x))


# dot function is used to compute inner products of vectors

# x = np.array([[1, 2], [3, 4]])
# y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# # Inner product of vectors
# print(np.dot(v, w))
# print(np.dot(w, v))
# print(v.dot(w))
# print(w.dot(v))
# print("------------\n")

# # Can use @ operator which is equivalent to numpy's dot product
# print(v @ w)
# print("------------\n")

# # Matrix / vector product
# print(x.dot(v))
# print(v.dot(x)) # order matter; get different output than above
# print(x @ v)

# print(v @ x)
# print(np.dot(x, v))
# print(np.dot(v, x))
# print("------------\n")

# # Matrix / matrix product
# print(x.dot(y))
# print(np.dot(x, y))
# print(x @ y)


##########
# numpy provides many functions to perform operations on array
x = np.array([[1, 2], [3, 4]])
y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# temp = np.sum(x) # Returns int
# print(temp, type(temp))
# print(np.sum(x)) # Compute sum of all elements
# temp = np.sum(x, axis=0)

# print(temp, type(temp))
# print(np.sum(x, axis=0)) # Compute sum for each column; returns array
# print(np.sum(x, axis=1)) # Compute sum for each row

#########
# # Transpose of an matrix
# print(x)
# print("transpose\n", x.T)
# print("------------")

# print(y)
# print("transpose\n", y.T) # transpose of array more than 2 rank seems complicated!! *_*


############
# Broadcasting
############

# # Broadcasting is a powerful mechanism that allows numpy to work with arrays of 
# # different shapes when performing arithmetic operations

# # We will add vector v to each row of matrix x, storing the result in matrix y
# x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# v = np.array([1, 0, 1])
# y = np.empty_like(x) # Create a empty matrix with the same shape as x

# print(x)
# print(v)
# print(y)
# print("-------")

# # Add the vector v to each row of matrix x with an explicit loop
# for i in range(4):
#   y[i, :] = x[i, :] + v

# print(y)

# # Above approach is slow for big matrices

# vv = np.tile(v, (4, 1)) # Stack 4 copies of v on top of each other
# print(vv)

# y = x + vv
# print(y)
# print("-------------\n")

# # numpy broadcasting allows us to perform this computation without creating multiple
# # copies of v

# y = x + v
# print(y)

###########
# Applications of broadcasting

# # Compute outer product of vectors
# v = np.array([1, 2, 3]) # v has shape (3,)
# w = np.array([4, 5]) # w has shape (2,)

# print(v)
# print(np.reshape(v, (3, 1))) # Reshaping v into a 3 X 1 column vector
# print(np.reshape(v, (3, 1)) * w)

# # Add a row to each row of a matrix
# x = np.array([[1, 2, 3], [4, 5, 6]]) # x has shape (2, 3)
# print(x + v) # v has shape (3,), so they broadcast to (2, 3)

# # Add a vector to each column of a matrix
# print((x.T + w).T)

# # Another solution is to reshape w to a row vector of shape (2, 1)
# print(x + np.reshape(w, (2, 1)))

# # Multiply a matrix by a constant
# # numpy treats arrays as a vector of shape ()
# # they can be broadcast together to shape (2, 3) producing the following array
# print(x * 2)



##########
# Matplotlib
##########

# Matplotlib is a plotting library
import matplotlib.pyplot as plt

# The most important function in matplotlib is plot, which allows us to plot 2D data

# Compute x and y cordinates for points on sine curve
# x = np.arange(0, 3 * np.pi, 0.1) # arange allows floating foints
# y = np.sin(x)

# # Plot the points using matplotlib
# # plt.plot(x, y)
# # print("-------\n")

# # Can plot multiple lines
# y_sin = np.sin(x)
# y_cos = np.cos(x)

# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel("x axis label")
# plt.ylabel("y axis label")
# plt.title("Sine and Cosine")
# plt.legend(["Sine", "Cosine"])


###########
# Subplot
###########

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1, and set the such subplot
# as active
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title("Sine")

# Set up a subplot grid that has height 2 and width 1, and set the such subplot
# as active
plt.subplot(2, 1, 2)

plt.plot(x, y_cos)
plt.title("Cosine")

# Show the figure
plt.show()
