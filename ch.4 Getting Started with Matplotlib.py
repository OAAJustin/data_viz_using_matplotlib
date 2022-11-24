## WHAT IS MATPLOTLIB?

# Matplotlib is the most popular visualization engine and other libraries such as Pandas and Seaborn rely on MPL for the actual visualizations.
# GeoPandas is another library, specialized for creating, manipulating and visualizing Geospatial data, based on Pandas, and thus, Matplotlib
# One of the key features of Matplotlib is the fact that it offers both a high-level API, that lets you easily plot most things,
# but also offers a granular, fully customizable low-level API if you'd like to tweak anything.
# Matplotlib supports two expression styles - MATLAB-style and Object-Oriented-Style

# The collection of methods that were made to achieve MATLAB-style, to a degree, inspired by MATLAB are brought together in the PyPlot interface

import matplotlib.pyplot as plt

# The functions we call from PyPlot, the interface of Matplotlib, all alter a figure and thus change its state.
# This state is saved and carried across function calls so calling multiple functions will essentially build on top of the state left previously.

x = [1, 2, 3, 4, 5]
y = [1, 4, 5, 7, 2]
plt.plot(x, y)
plt.show()

# Call plt.plot() function again, this time, on the same X-axis, but setting new values on the Y-axis.

x = [1, 2, 3, 4, 5]
y = [1, 4, 5, 7, 2]
z = [1, 6, 2, 5, 1]
plt.plot(x, y)
plt.plot(x, z)
plt.show()

# We can also plot two different lines, on the same figure, with totally different X-values

x_1 = [1, 2, 3, 4, 5]
y_1 = [1, 4, 5, 7, 2]
x_2 = [6, 7, 8, 9, 15]
y_2 = [1, 6, 2, 5, 1]

plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
plt.show()

# We can make the X-axis of one line numerical, while the X-axis of the other line is categorical. These will need to be the same shape.

x_1 = [1, 2, 3, 4, 5]
y_1 = [1, 4, 5, 7, 2]
x_2 = ['a', 'b', 'c', 'd', 'e']
y_2 = [1, 6, 2, 5, 1]

plt.plot(x_1, y_1)
plt.plot(x_2, y_2, 'ro')
plt.show()

# Categorical values and numerical values clash, the ones plotted the latest prevail.
# r - Color (r is red, g is green, b is blue)
# o - Shape (o is circle, - is line...)
# The default blue line we see maps to the b- argument, while we've also seen o- (orange line).

# Using PyPlot instance, we can access and customize various aspects of the plots, such as labels.

x_1 = [1, 2, 3, 4, 5]
y_1 = [1, 4, 5, 7, 2]
x_2 = ['a', 'b', 'c', 'd', 'e']
y_2 = [1, 6, 2, 5, 1]

plt.plot(x_1, y_1)
plt.plot(x_2, y_2, 'ro')
plt.ylabel('Y-Axis Label')
plt.xlabel('X-Axis Label')
plt.show()

## ANATOMY OF MATPLOTLIB PLOTS

# Figure - The figure that contains everything that we'll be seeing within it. Each figure object can have one or more Axes objects
# Axes - Although the name Axes implies the actual axes of the plot, the Axes object can practically be seen as the plot itself.
    # An Axes sits snug in the Figure and contains elements such as Titles, Legends, Grids, etc.
    # Since a Figure can have multiple Axes objects, each would actually  be a plot for itself.
# Title - The title of the Axes object
# Legend - The legend of the Axes object.
# Ticks - Sub-divided into major ticks and minor ticks.
# Labels - Labels can be set for the X and Y-axis, or for the ticks.
# Grids - Optional lines in the background fo the plot, that help the interpreter to distinguish between similar X and Y values,
    # based on the frequency of grid lines.
# Lines/Markers - The actual lines/markers that are used to express records/data of a plot.
    # Most of the time, you'll use lines to plot continuous data, while you'll use markers for discrete data.

## OBJECT-ORIENTED PLOTTING

figure = plt.figure()
ax = figure.add_axes([0, 0, 1, 1])

x = [1, 2, 3, 4, 5]
y = [1, 7, 3, 9, 3]

ax.plot(x, y)
plt.show()

# AX object holds information related to the position of the figure. Respectively, left, bottom, width and height
# This is bare bones and probably wouldn't be used commonly - A more popular option is the add_subplots() that adds a subplot(Axes),
    # or even the subplots() function that can be used to add multiple subplots at once.

figure = plt.figure()
ax = figure.add_subplot(111)

x = [1, 2, 3, 4, 5]
y = [1, 7, 3, 9, 3]

ax.plot(x, y)
plt.show()

# Here, we're specifying in which position on the grid we want our Axes to be added to.
# Additionally, when calling this method, we construct the grid.
# Not only supplying the index of the Axes but also letting Matplotlib know how many columns and rows the grid will have.

# 1 - Grid has one row (X Axis)
# 1 - Grid has one column (Y Axis)
# 1 - Index of the subplot we're adding

# Set the position to 122

figure = plt.figure()
ax = figure.add_subplot(122)

x = [1, 2, 3, 4, 5]
y = [1, 7, 3, 9, 3]

ax.plot(x, y)
plt.show()

# Populate the position on 121

figure = plt.figure()
ax = figure.add_subplot(122)
ax2 = figure.add_subplot(121)

x = [1, 2, 3, 4, 5]
y = [1, 7, 3, 9, 3]
z = [1, 6, 2, 4, 5]

ax.plot(x, y)
ax2.plot(x,z)
plt.show()

# Alternatively, you can use the subplots() function instead, to create one or multiple subplots, as well as the Figure object.
# This is a popular way of instantiating them, since you can do it all in one line.

fig, ax = plt.subplots()
x = [1, 2, 3, 4, 5]
y = [1, 7, 3, 9, 3]

ax.plot(x, y)
plt.show()

# If you'd like to work with more than one subplot, you simply chuck in the number of them into the subplots() call.
# Create 4 Axes objects, and plot 4 different plots on a single figure.

fig, ax = plt.subplots(4)

x = [5, 4, 2, 6, 2]
y = [1, 7, 3, 9, 3]
z = [1, 6, 2, 4, 5]
n = [7, 3, 2, 5, 2]

ax[0].plot(x)
ax[1].plot(y)
ax[2].plot(z)
ax[3].plot(n)

plt.show()

# The returned ax is a Numpy array of Axes objects. We can access them each through the usual notation array[index]
# This would allow you to use a for loop to access all Axes of a Figure, incrementing the index on each call.

for i in range(4):
    ax[i].plot(x)

# Otherwise, if you want to have them as separate objects, each with a unique reference, you can:

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

x = [5, 4, 2, 6, 2]
y = [1, 7, 3, 9, 3]
z = [1, 6, 2, 4, 5]
n = [7, 3, 2, 5, 2]

ax1.plot(x)
ax2.plot(y)
ax3.plot(z)
ax4.plot(n)

plt.show()

## PYPLOT API VS. OBJECT-ORIENTED API

# Object-Oriented API has a more convenient customization approach, since you cna directly work with objects and their parameters.
# The PyPlot API autoconfigures a lot of the commonly used options, so you don't have to worry about them - to a degree, it abstracts the
    # inner-workings of Matplotlib, without locking you out from accessing the low-level parameters.
# You'll use both of these pretty much interchangeably, all the time. You can mix both paradigms - plot on a Figure using the plt instance,
    # and then extract the Axes and Figure objects from it.

x_1 = [1, 2, 3, 4, 5]
y_1 = [1, 4, 5, 7, 2]
y_2 = [6, 3, 5, 3, 2]

plt.plot(x_1, y_1)

# Get current 'Axes'
ax = plt.gca()
# Get current 'Figure'
fig = plt.gcf()

ax.plot(x_1, y_2)

plt.show()

# When accessing elements, such as labels, titles, figure sizes, etc - there are two access conventions.
# In PyPlot's API, you directly access them, like how you'd access xlabel() for example.
# In the Object-Oriented API, you can access them through getter and setter methods, such as set_xlabel().

# PyPlot API

"""
plt.ylabel('Y-Axis Label')
plt.xlabel('X-Axis Label')
"""

# Object-Oriented API using setter functions

"""
ax.set_ylabel('Y-Axis Label')
ax.set_xlabel('X-Axis Label')
"""

# Object-Oriented API has two ways to use setters:

# The generic set() function
# Specific set_x() functions

# The former allows you to write one line for multiple arguments while the latter provides you with the code that's more readable if you change
    # a large number of parameters.

"""
as.set(xlabel = 'X-Axis Label', ylabel = 'Y-Axis label')
OR
ax.set_ylabel('Y-Axis Label')
ax.set_xlabel('X-Axis Label')

"""