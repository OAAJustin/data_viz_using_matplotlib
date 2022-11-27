## CHANGING THE FIGURE SIZE
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import random


# Re-create the plot that couldn't fit very well into the figure.

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

# SETTING THE FIGSIZE ARGUMENT

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (6, 8))

x = [5, 4, 2, 6, 2]
y = [1, 7, 3, 9, 3]
z = [1, 6, 2, 4, 5]
n = [7, 3, 2, 5, 2]

ax1.plot(x)
ax2.plot(y)
ax3.plot(z)
ax4.plot(n)

plt.show()

# The size is defined in inches, not pixels, which is a fairly intuitive way to imagine the size of plots.
# Matplotlib doesn't currently support the metric scale, though, it's easy to write a helper function to convert between the two:

def cm_to_inches(value):
    return value / 2.54

# Now you can adjust the size of the plot by using the function to convert any values you put in, into inches.

plt.figure(figsize = (cm_to_inches(15), cm_to_inches(10)))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# Alternatively, if you're creating a Figure object for your plot separately from your Axes object(s), you can assign the size at that time.

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure(figsize = (8,6))

ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.plot(x, y)
ax2.plot(x, z)
plt.show()

# SETTING THE HEIGHT AND WIDTH OF A FIGURE IN MATPLOTLIB

# Can be done either via the set() function with the figheight and figwidth argument, or via the set_figheight() and set_figwidth() functions.

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure()

fig.set_figheight(5)
fig.set_figwidth(10)

ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.plot(x, y)
ax2.plot(x, z)
plt.show()

## UNDERSTANDING MATPLOTLIB TEXT (TITLES, LABELS, ANNOTATIONS)

# Some of the text-related commands you'll be using to add text like labels and titles to your plots are:

"""
plt.text() or ax.text()
plt.annotate() or ax.annotate()
plt.xlabel() / plt.ylabel or ax.set_xlabel() / ax.set_ylabel()
plt.title() or ax.set)title()
plt.suptitle() or fig.suptitle()
"""

# Generate  blank plot and add some text:

fig, ax = plt.subplots()

fig.suptitle('This is the Figure-level Suptitle')
ax.set_title('This is the Axes-level Title')
ax.set_xlabel('X-Label')
ax.set_ylabel('Y-Label')
ax.text(0.5, 0.5, 'This is generic text')
ax.annotate('This is an annotation, with an arrow between \n itself and generic text',
            xy = (0.625, 0.5),
            xytext = (0.25, 0.25),
            arrowprops = dict(arrowstyle = '<->',
                            connectionstyle = 'arc3, rad = 0.15'))

plt.show()

# Suptitle() is added at the Figure-level, and is above all of its subplots.
# The title and labels can be set on the Axes-level, where each Axes can have separate titles and labels.
# Generic text() method is supplied positional x and y value arguments.
# Keep in mind that x and y here refer to the actual values on the plots - not percentages.
# Their positions will depend on the data you're plotting and the scale they are in.
# By default, Matplotlib creates a 1x1 plot ( the X-axis has 0..1 values, and the Y-axis has 0..1 values as well)
# annotate() method accepts several arguments such as teh xy tuple, xytext tuple and arrowprops.
# The xy tuple is the end-point of the annotation - to what it's pointing.
# The xy text tuple specifies where the text of the annotation will be positioned
# NOTE: Text instances support the use of a newline (\n) to break text into new lines.

# Annotations are super useful when it comes to pointing things out on a plot.
# The arrowprops dictionary is where you customize them and you can set various properties

"""
width - with of the arrow
headwidth - width of the head of the arrow
shrink - shrinking the arrow to allow for space between the annotation text and annotated point
boxstyle - allows you to set the box style of the arrows
arrowstyle - allows you to choose between types of arrows
connectionstyle - allows you to set the connection style.
"""

# Plot a figure and add several annotations, with varying styles and types

fig, ax = plt.subplots()

fig.suptitle('Different Types of Annotations')

ax.annotate('Typical annotation',
            xy = (0.8, 0.1),
            xytext = (0.1, 0.1),
            arrowprops = dict(facecolor = 'black', width = 1, headwidth = 10),
            verticalalignment = 'center')

ax.annotate('Annotation with arrow style',
            xy = (0.8, 0.2),
            xytext = (0.1, 0.2),
            arrowprops = dict(arrowstyle = '->'),
            verticalalignment = 'center')

ax.annotate('Annotation with arrow style',
            xy = (0.8, 0.3),
            xytext = (0.1, 0.3),
            arrowprops = dict(arrowstyle = '<->'),
            verticalalignment = 'center')

ax.annotate('Annotation with arrow style',
            xy = (0.8, 0.4),
            xytext = (0.1, 0.4),
            arrowprops = dict(arrowstyle = '|-|'),
            verticalalignment = 'center')

ax.annotate('Annotation with arrow style',
            xy = (0.8, 0.5),
            xytext = (0.1, 0.5),
            arrowprops = dict(arrowstyle = 'fancy'),
            verticalalignment = 'center')

plt.show()

# verticalalignment argument ahs been set to center so the arrows get aligned with the center line of the annotation text.
# You can set it to top and bottom as well, though the arrows will be tilted upward or downward depending on the font size.

fig, ax = plt.subplots()

fig.suptitle('Different Types of Annotations')

# X - Axis values
x = [1, 2]
# Y - Axis values, empty for X=1, with several values for X=2
y = [(),(1, 2, 3, 4)]

for xe, ye in zip(x, y):
    ax.scatter([xe] * len(ye), ye)

plt.xlim(0, 2)

ax.annotate('Connection style with Angle',
            xy = (2, 1),
            xytext = (0, 4),
            arrowprops = dict(arrowstyle = '->',
                                connectionstyle = 'angle, angleA = 120, angleB = 0'),
            verticalalignment = 'center')

ax.annotate('Connection style with Angle3',
            xy = (2, 3),
            xytext = (0, 2),
            arrowprops = dict(arrowstyle = '->',
                                connectionstyle = 'angle3, angleA = 90, angleB = 0'),
            verticalalignment = 'center')

ax.annotate('Connection style with Arc',
            xy = (2, 2),
            xytext = (0,3),
            arrowprops = dict(arrowstyle = '->',
                                connectionstyle = 'arc, angleA = 0, angleB = 0, armA = 0, armB = 45'),
            horizontalalignment = 'left')

ax.annotate('Connection style with Arc3',
            xy = (2, 4),
            xytext = (0,1),
            arrowprops = dict(arrowstyle = '->',
                                connectionstyle = 'arc3, rad = 0.25'),
            horizontalalignment = 'left')

plt.show()

# There are four main styles of connectionstyle you can apply: angle, angle3, arc, arc3

"""
Name; Possible Attributes; Class
angle; angle3; arc; arc3
angleA = 90, angleB = 0, rad = 0.0; angleA = 90, angleB = 0; angleA = 0, angleB = 0, armA = None, armB = None, rad = 0.0; rad = 0.0
Angle; Angle3; Arc; Arc3
"""

# The Angle is used to create a quadratic Bezier path between the points referenced to by xy and xytext.
# The break-point is placed at the intersection of the lines that point to the start and end point, each at a certain slope - angleA and angleB.
# The optional rad argument rounds the edge(s) created by the break
# Angle3 is used for the same purpose - though, it has 3 control points.
# Styles: fancy, simple and wedge
# They can only work with Angle3 and Arc3 connection style variants
# Arc allows to have two additional arms at the start and end of the path. Can be used to introduce new breaking angles that already exist.
# The optional rad argument rounds the edges off.
# Arc3 creates a regular, simple, Bezier curve between two points and the middle control point is placed in the middle of the path.
# The rad argument is used to calculate the distance between this central control point and the straight line between the starting  and end point.
# The higher the rad value is, the more further away the central point of the curve will be - in effect, curving it more.

## ADD LEGEND

# Annotations are typically used to point out a certain observation, rather than entire features.
# A legend, which has a list of colors, and a list of labels for those colors are typically used.

# Create a plot with two variables, each with a different color.

fig, ax = plt.subplots()

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue')
ax.plot(z, color = 'black')

plt.show()

# Create a legend

fix, ax = plt.subplots()

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
leg = ax.legend()

plt.show()

# Once labeled, call the legend() function on the Axes instance.

## CUSTOMIZING A LEGEND IN MATPLOTLIB

# Remove the border around the legend and place it at the top-right corner of the plot.

fig, ax = plt.subplots(figsize = (12, 6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
leg = ax.legend(loc = 'upper right', frameon = False)

plt.show()

# loc argument is used to specify the location of a legend, intuitively designed - a combination of upper, lower, or
    # center with left, right or center. center alone puts the legend in the center of the plot, naturally.
# By leaving this field empty matplotlib will do its best (by default) to fit the legend accurately.

## ADDING A LEGEND OUTSIDE OF AXES

# Legends can be placed outside of the axes, and away from the elements that constitute it via bbox_to_anchor argument to specify the anchor.

fig, ax = plt.subplots(figsize = (12, 6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
leg = ax.legend(loc = 'center', bbox_to_anchor = (0.5, -0.10), shadow = False, ncol = 2)

plt.show()

# bbox_to_anchor argument accepts a few arguments itself. Firstly, a tuple, which allows up to 4 elements - x, y, width, height
# shadow has been set to False - this is used to specify whether we want a small shadow rendered below the legend or not
# ncol argument has been set to 2 - this is to specify the number of labels in a column.
# if this argument is changed to 1, the labels would be placed one above the other.

fig, ax = plt.subplots(figsize = (12, 6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
leg = ax.legend(loc = 'center', bbox_to_anchor = (0.5, -0.10), shadow = False, ncol = 1)

plt.show()

# NOTE: The bbox_to_anchor argument is used alongside the loc argument. The loc argument will put the legend based on the bbox_to_anchor.

## CHANGING THE LEGEND SIZE

# Legends scale by the size of the text within them. Changing the font size of the text within will, therefore, change the legend size as well.
# Pass the fontsize argument when creating a legend.

fig, ax = plt.subplots(figsize = (12, 6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
leg = ax.legend(title = 'Functions', fontsize = 12, title_fontsize = 14,
                loc = 'center', bbox_to_anchor = (0.5, -0.10), shadow = False, ncol = 2)

plt.show()

## CHANGING THE FONT SIZE

# Add X and Y labels to the plot as well as title to the Axes

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
ax.set_title('Sine and Cosine Waves')
ax.set_xlabel('Time')
ax.set_ylabel('Intensity')
leg = ax.legend()

plt.show()

## CHANGING THE FONT SIZE USING fontsize

# Every function that deals with text, such as Title, labels and all other textual functions accepts an argument - fontsize

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
ax.set_title('Sine and Cosine Waves', fontsize = 20)
ax.set_xlabel('Time', fontsize = 16)
ax.set_ylabel('Intensity', fontsize = 16)
leg = ax.legend()

plt.show()

# We can also change the size of the font in elements by passing in the prop argument and assigning a font size value to the "size" parameter
# Though, a more high-level and user-friendly approach is by simply using the fontsize argument.
# However, while we can set each font size like this, if we have many textual elements, and just want a uniform, general size - set font globally

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
ax.set_title('Sine and Cosine Waves')
ax.set_xlabel('Time')
ax.set_ylabel('Intensity')
leg = ax.legend(prop = {'size': 16})

plt.show()

## CHANGING THE FONT SIZE GLOBALLY

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

plt.rcParams['font.size'] = '16'

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
ax.set_title('Sine and Cosine Waves')
ax.set_xlabel('Time')
ax.set_ylabel('Intensity')
leg = ax.legend()

plt.show()

# You must set these before the plot() function call since if you try to apply runtime configurations afterwards, no changes will be made.
# X and Y ticks, nor the X and Y labels have changed in size. You won't be able to change these by changing font group.
# You'd use axes.labelsize and xtick.labelsize/ytick.labelsize for them respectively - these params belong to xtick and ytick groups.

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

plt.rcParams['font.size'] = '16'
plt.rcParams['xtick.labelsize'] = '16'
plt.rcParams['ytick.labelsize'] = '16'

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
ax.set_title('Sine and Cosine Waves')
ax.set_xlabel('Time', fontsize = 16)
ax.set_ylabel('Intensity', fontsize = 16)
leg = ax.legend()

plt.show()

# SAVE PLOT AS IMAGE

# Save by either clicking the "Save" button, denoted by a floppy disk on the window opened by plt.show() or programmatically by using savefig()

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')
ax.set_xlabel('Time', fontsize = 16)
ax.set_ylabel('Intensity', fontsize = 16)

fig.suptitle('Sine and Cosine Waves')
leg = ax.legend()

plt.savefig('saved_figure')

# NOTE: savefig() function isn't uniquie to the plt instance. You can use it on a Figure instance (but not the Axes instance) as well.

fig.savefig('Saved_figure')

# The savefig() function accepts a mandatory filename argument. Additionally, it accepts other options, such as dpi, transparent, bbox_inches, etc

## SETTING THE IMAGE DPI

# The DPI parameter defines the number of dots(pixels) per inch. This is essentially the resolution of the image we're producing.
# A higher DPI can be used to produce larger files, with higher resolution (for larger screens) where resolution is important

fig = plt.figure()

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)
fig.savefig('saved_figure-50dpi.png', dpi = 50)
fig.savefig('saved_figure-100dpi.png', dpi = 100)
fig.savefig('saved_figure-1000dpi.png', dpi = 1000)

# default value is 100

## SAVE TRANSPAREWNT IMMAGE WITH MATPLOTLIB

# The transparent argument can be used to create a plot with a transparent background. All elements will be visible, but the background
    # will be transparent, allowing any underlying pattern or color to be seen

fig = plt.figure()

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)

fig.savefig('saved_figure-transparent.png', transparent = True)

## CHANGING PLOT COLORS

# Set a color for the facecolor argument - it accepts a color and defaults to white
# Change plot face color to red

fig = plt.figure()

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)

fig.savefig('saved_figure-colored.png', facecolor = 'red')

## SETTING IMAGE BORDER BOX

# bbox_inches argument accepts a string and specifies the border around the box we're plotting.
# Crop around the box as much as possible by setting the bbox_inches arugment to 'tight':

fig = plt.figure()

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)

fig.savefig('saved_figure-colored.png', bbox_inches = 'tight', facecolor = 'red')

# By default, bbox for your plot encompasses the entire plot, with some padding on each side.
# Padding was reduced when bbox_to_inches was set to 'tight'
# You may also set bbox to any arbitrary value in inches by setting its starting and ending coordinates
# Set the bbox to start at the 1-inch mark on the X-axis, and 1-inch mark on the Y-axis.
# Then, to end at the 2-inch mark on the X-axis, and 2-inch mark on the Y-axis

fig = plt.figure()

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)
fig.savefig('saved_figure-bbox.png', bbox_inches = matplotlib.transforms.Bbox([[1, 1],[2, 2]]), facecolor = 'blue')

## SET AXIS RANGE (xlim, ylim)

# We can crop and focus on certain parts of plots by setting the axis range - also knows as X-limit and Y-limit
# We can also expand the range of data by setting a higher range than we have data for.
# Matplotlib automatically sets the axis range based on the provided data

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')

plt.show()

## SETTING THE X-LIMIT (XLIM) FOR AXES

# # Set the X-0limit, using either PyPlot and Axes instances - both methods accept a tuple -= the left and right limits.

# Show truncated view to only show the dat in range of 25-50 on the X-axis

" xlim([25, 50]) "

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')

plt.xlim([25, 50])
plt.show()

# The same effect can be achieved by setting these via the ax object. If we had multiple Axes, we can set the limit for them separately

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax.set_title('Full View')
ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')

ax2.set_title('Truncated View')
ax2.plot(y, color = 'blue', label = 'Sine Wave')
ax2.plot(z, color = 'black', label = 'Cosine Wave')

ax2.set_xlim([25, 50])

plt.show()

## SETTING THE Y-LIMIT (ylim) FOR AXES

# Same 2 approaches

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')

plt.ylim([-1, 0])
plt.show()

# OR

fig, ax = plt.subplots(figsize = (12,6))

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

ax.plot(y, color = 'blue', label = 'Sine Wave')
ax.plot(z, color = 'black', label = 'Cosine Wave')

ax.set_ylim([-1, 0])
plt.show()

## CHANGE TICK FREQUENCY

# Create plot with random set of numbers

fig = plt.subplots(figsize = (12, 6))

x = np.random.randint(low = 0, high = 50, size = 100)
y = np.random.randint(low = 0, high = 50, size = 100)

plt.plot(x, color = 'blue')
plt.plot(y, color = 'red')

plt.show()

## SEETTING FIGURE-LEVEL TICK FREQUENCY

# You can set the tick frequency on a Figure-level and Axes-level.

fig = plt.subplots(figsize = (12, 6))

x = np.random.randint(low = 0, high = 50, size = 100)
y = np.random.randint(low = 0, high = 50, size = 100)

plt.plot(x, color = 'blue')
plt.plot(y, color = 'red')

plt.xticks(np.arange(0, len(x) + 1, 5))
plt.yticks(np.arange(0, max(y), 2))

plt.show()

# You can use the xticks() and yticks() functions and pass in an array denoting the actual ticks.
# Start, Stop, Step
# Note that the yticks() and xticks() functions don't change the frequency. They set the ticks, and the step argument helps display the f-tick

## SETTING AXIS-LEVEL TICK FREQUENCY

# IF you have multiple plots, you may want to change the tick frequency on the Axes-level.
# Use the set_xticks() and set_yticks() functions on the returned Axes instance when adding subplots to a Figure.
# Create a Figure with two axes and change the tick frequency on them separately.

fig = plt.figure(figsize = (12, 6))

ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


x = np.random.randint(low = 0, high = 50, size = 100)
y = np.random.randint(low = 0, high = 50, size = 100)
z = np.random.randint(low = 0, high = 50, size = 100)

ax.plot(x, color = 'blue')
ax.plot(y, color = 'red')
ax2.plot(y, color = 'red')
ax2.plot(z, color = 'green')

ax.set_xticks(np.arange(0, len(x)+1, 5))
ax.set_yticks(np.arange(0, max(y), 2))
ax2.set_xticks(np.arange(0, len(x)+1, 25))
ax2.set_yticks(np.arange(0, max(y), 25))

plt.show()

## ROTATE AXIS TICK LABELS

# Plot a figure that visualizes a variable against dates

fig, ax = plt.subplots(figsize = (8,4))

# [Timestamp('2022-11-26 00:22:01:652993', freq = 'D')
x = pd.date_range(datetime.today(), periods = 10).tolist()
# [1, 7, 5, 4, 2, 6, 9, 0, 3, 8]
y = random.sample(range(0, 10), 10)

ax.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Intensity')

plt.show()

## ROTATING X-AXIS TICK LABELS

# Rotate Figure-level using plt.xticks() or rate tehm on an Axes-level by using tick.set_rotation() individually
# Or by using ax.set_xticklabels() and ax.xtick_params()

fig, ax = plt.subplots(figsize = (8,4))

x = pd.date_range(datetime.today(), periods = 10).tolist()
y = random.sample(range(0, 10), 10)

ax.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.xticks(rotation = 15)

plt.show()

# rotation of xticks has been set to 15, to signify a 15-degree tilt, counter-clockwise
# Alternatively, iterate over the ticks in the ax.get_xticklabels() list, then, call tick.set_rotation() on each

fig, ax = plt.subplots(figsize = (8,4))
x = pd.date_range(datetime.today(), periods = 10).tolist()
y = random.sample(range(0, 10), 10)

ax.plot(x, y)
plt.draw()

for tick in ax.get_xticklabels():
    tick.set_rotation(15)

plt.xlabel('Time')
plt.ylabel('Intensity')

plt.show()

# NOTE: You will need to call plt.draw() before accessing or setting the X-tick labels because the labels are populated after the plot is drawn.
# Otherwise, they'll return empty text values
# Finally, you can use the ax.tick_params() function and set the label rotation there


fig, ax = plt.subplots(figsize = (8,4))
x = pd.date_range(datetime.today(), periods = 10).tolist()
y = random.sample(range(0, 10), 10)

ax.plot(x, y)
ax.tick_params(axis = 'x', labelrotation = 15)

plt.xlabel('Time')
plt.ylabel('Intensity')

plt.show()

## ROTATE Y-AXIS TICK LABELS

# All steps can be applied to Y-axis as X-axis
# Figure-level with plt.yticks(), or on the Axes-label by using tick.set_rotation() or by manipulating ax.tick_params()

fig, ax = plt.subplots(figsize = (8,4))
x = pd.date_range(datetime.today(), periods = 10).tolist()
y = random.sample(range(0, 10), 10)

ax.plot(x, y)

# Approach 1

plt.yticks(rotation = 15)
plt.xticks(rotation = 15)

# Approach 2

plt.draw()

for tick in ax.get_yticklabels():
    tick.set_rotation(15)

# Approach 3

ax.tick_params(axis = 'y', labelrotation = 15)

plt.xlabel('Time')
plt.ylabel('Intensity')
plt.show()

## ROTATE DATES TO FIT AUTOMATICALLY

# An even easier method than previously is using fig.autofmt__date(). This can either be used as fig.autofmt_xdate() or fig.autofmt_ydate().

file = 'seattleWeather.csv'
weather_data = pd.read_csv(file)

fig = plt.figure()
plt.plot(weather_data['DATE'], weather_data['PRCP'])
fig.autofmt_xdate()
plt.show()

## DRAW VERTICAL LINES ON PLOT

fig, ax = plt.subplots(figsize = (8, 4))

dates = pd.date_range(datetime.today(), periods = 15).to_list()
y_1 = [1, 2, 4, 3, 5, 5, 6, 7, 5, 6, 7, 8, 9, 9, 10]
y_2 = [1, 1, 3, 2, 4, 4, 5, 5, 4, 5, 6, 7, 10, 12, 13]

ax.plot(dates, y_1, label = 'Competitor Company')
ax.plot(dates, y_2, label = 'Our Company')

plt.xticks(rotation = 15)

plt.ylabel('Sales')
plt.legend()
plt.show()

# There are two ways you can draw lines, using the vlines9) or axvline() functions of the PyPlot instance. You can also call methods on Axes obj.

fig, ax = plt.subplots(figsize = (8, 4))

dates = pd.date_range(datetime.today(), periods = 15).to_list()
y_1 = [1, 2, 4, 3, 5, 5, 6, 7, 5, 6, 7, 8, 9, 9, 10]
y_2 = [1, 1, 3, 2, 4, 4, 5, 5, 4, 5, 6, 7, 10, 12, 13]

ax.plot(dates, y_1, label = 'Competitor Company')
ax.plot(dates, y_2, label = 'Our Company')

ax.vlines(['2022-11-29', '2022-12-09'], 0, 15, linestyles = 'dashed', colors = 'red') # NOTE: Remember for this example, we use today's date

plt.xticks(rotation = 15)

plt.ylabel('Sales')
plt.legend()
plt.show()

# The vlines() function accepts a few arguments - a scalar, or 1D array of X-values that you'd like to draw on.
# Then the ymin and ymax arguments - these have been set from 0 to max(y_2), since we don't want the lines to go higher or lower than max plot value
# Then styles such as linestyles or colors, which accept the typlical Matplotlib styling options.
# This function would allow us to set the ymin and ymax in concrete values, while axvline() lets us choose the height percentage-wise.
# The bottom to the top is set by default.

fig, ax = plt.subplots(figsize = (8, 4))

dates = pd.date_range(datetime.today(), periods = 15).to_list()
y_1 = [1, 2, 4, 3, 5, 5, 6, 7, 5, 6, 7, 8, 9, 9, 10]
y_2 = [1, 1, 3, 2, 4, 4, 5, 5, 4, 5, 6, 7, 10, 12, 13]

ax.plot(dates, y_1, label = 'Competitor Company')
ax.plot(dates, y_2, label = 'Our Company')

ax.set_ylim(-25, 50)
ax.vlines(['2022-11-29'], -10, 15, linestyles = 'dashed', colors = 'red')
ax.vlines(['2022-12-09'], -5, 20, linestyles = 'dashed', colors = 'red')

plt.xticks(rotation = 15)

plt.ylabel('Sales')
plt.legend()
plt.show()

# We can set multiple colors for these lines, by passing in a list of colors instead of one.

fig, ax = plt.subplots(figsize = (8, 4))

dates = pd.date_range(datetime.today(), periods = 15).to_list()
y_1 = [1, 2, 4, 3, 5, 5, 6, 7, 5, 6, 7, 8, 9, 9, 10]
y_2 = [1, 1, 3, 2, 4, 4, 5, 5, 4, 5, 6, 7, 10, 12, 13]

ax.plot(dates, y_1, label = 'Competitor Company')
ax.plot(dates, y_2, label = 'Our Company')

ax.vlines(['2022-11-29', '2022-12-09'], 0, max(y_2), linestyles = 'dashed', colors = ['red', 'blue'])

plt.xticks(rotation = 15)

plt.ylabel('Sales')
plt.legend()
plt.show()

## DRAWING VERTICAL LINES WITH PYPLOT.AXVLINE()

# axvline() accepts a single X-axis variable t draw a line on.
# We're working with <class 'pandaas.libs.tslibls.timestamps.Timestamp'> objects - not datetime objects here
# Matplotlib can't convert between these automatically. We can use Pandas' to_datetime() function to convert a string to another
# <class 'pandas._libs.tslibs.timestamps.Timestamp'> explicitly, which Matplotlib uses as our X-axis point to draw a vertical line on
# Alternatively, you can put the Timestamp in a list, which resolves this conversion issue.

fig, ax = plt.subplots(figsize = (8, 4))

dates = pd.date_range(datetime.today(), periods = 15).to_list()
y_1 = [1, 2, 4, 3, 5, 5, 6, 7, 5, 6, 7, 8, 9, 9, 10]
y_2 = [1, 1, 3, 2, 4, 4, 5, 5, 4, 5, 6, 7, 10, 12, 13]

ax.plot(dates, y_1, label = 'Competitor Company')
ax.plot(dates, y_2, label = 'Our Company')

ax.axvline(pd.to_datetime('2022-11-29'), color = 'red')
ax.axvline(pd.to_datetime('2022-12-09'), color = 'blue')

# OR

ax.axvline(['2022-11-29'], color = 'red')
ax.axvline(['2022-12-09'], color = 'blue')

plt.xticks(rotation = 15)

plt.ylabel('Sales')
plt.legend()
plt.show()

# Being able to only plot on a single point at a time means that if we want to plot on multiple points, we have to call mutliple times.
# This method does not let us specify the linestyle like vlines(), though, it doesn't require the ymin and ymax arguments by default.
# if omitted, they'll simply be form the top to the bottom of the Axes.
# You may change the height - this time around in percentages.
# These percentages take the top and bottom of the Axes into consideration so 0% will be at the very bottom, while 100% will be at the very top.

# Draw the first line as spanning from 5% to 40%, and the second line as spanning from 50% to 90%

fig, ax = plt.subplots(figsize = (8, 4))

dates = pd.date_range(datetime.today(), periods = 15).to_list()
y_1 = [1, 2, 4, 3, 5, 5, 6, 7, 5, 6, 7, 8, 9, 9, 10]
y_2 = [1, 1, 3, 2, 4, 4, 5, 5, 4, 5, 6, 7, 10, 12, 13]

ax.plot(dates, y_1, label = 'Competitor Company')
ax.plot(dates, y_2, label = 'Our Company')

ax.axvline(pd.to_datetime('2022-11-29'), 0.4, 0.05, color = 'red')
ax.axvline(pd.to_datetime('2022-12-09'), 0.9, 0.50, color = 'blue')

plt.xticks(rotation = 15)

plt.ylabel('Sales')
plt.legend()
plt.show()