import pandas as pd
import numpy as np

# A series is similar to a one-dimensional array. It can store data of any type.
# The values of a Pandas Series are mutable but the size of a Series is immutable and cannot be changed. 

# First element is assigned the index of 0 while the last element is at index N-1, where N is the total number of elements in the series. 

## CREATING SERIES FROM PYTHON ARRAY

# PYTHON ARRAYS
data = [0, 1, 2, 3, 4]
index = [1, 2, 3, 4, 5]

series = pd.Series(data, index)

## CREATING SERIES FROM PYTHON DICTIONARY

series_2 = pd.Series({"KEY 1": 1, "KEY 2": 2, "KEY 3": 3}, 
                        index = ["KEY 3", "KEY 1", "KEY 2", "INVALID KEY"])

# The order of the dictionary is reflected in the populated Series, so if we jumbled the keys around in the dictionary,
# that order would be preserved.

# When a key contains no value, Pandas default value mapped would be NaN (Not a Number).

## CREATING SERIES FROM NUMPY ARRAY

fruits = np.array(['apple', 'orange', 'mango', 'pear'])
series_fruits = pd.Series(fruits)


## ACCESSING SERIES ELEMENTS

element = series_fruits[0]

# Slice Notation

fruits_extended = np.array(['apple', 'orange', 'mango', 'pear', 'kiwi', 'pineapple'])
series_fruits_extended = pd.Series(fruits_extended, index = [7, 4, 6, 2, 5, 0])

elements = series_fruits[2:5]

# It is worth noting that elements is also a Series. Use slice notation to access sub-Series

first_two_elements = series_fruits_extended[:2]
last_two_elements = series_fruits_extended[-2:]

## ACCESSING CUSTOM INDICES

# If you'd like to access elements, based on their index instead of position, you can simply pass in one index

element5 = series_fruits_extended[5]

# You may use different data types to map for labels, which can be accessed as well.

decimal_values = [80, 121, 116, 104, 111, 110]
series_decimal = pd.Series(decimal_values, index = ["P", "y", "t", "h", "o", "n"])

element_decimal1 = series_decimal['y']
element_decimal2 = series_decimal[4]

# Searching for elements individually is inefficient, you may search for multiple elements by separating them by a comma.

elements_decimal_by_position = series_decimal[[1,4]]
elements_decimal_by_labels = series_decimal[['P', 'n']]

## SERIES.HEAD()

# The head() method prints the first 5 elements, by default, and is most commonly used to print out the first few values
# to verify that it's been constructed correctly, or just take a peek at the data inside.
# The method also accepts an optional argument, which you can use to specify how many elements you'd like to print.

head_series_decimal_10 = series_decimal.head(10)

## SERIES.TAIL()

# The tail() method does the exact opposite of head() - it prints the last n elements of a Series, and n=5 by default.

tail_series_decimal_3 = series_decimal.tail(3)

## DATAFRAMES

# While Series' are essentially one-dimensional labeled arrays of any type of data, DataFrames are two-dimensional, heterogeneous,
# labeled arrays of any type of data. (Heterogeneous - not all "rows" need to be of equal size)

## CREATING DATAFRAMES

# DataFrames must be chronological in tablature.

## CREATING AN EMPTY DATAFRAME

pepper_list = [
    [50, "Bell pepper", "Not even spicy"],
    [5000, "Espelette pepper", "Uncomfortable"],
    [500000, "Chocolate habero", "Practically ate pepper spray"]
]

dataframe = pd.DataFrame(pepper_list)

# The same effect could've been achieved by having the data in multiple lists and zip()-ing them together

scoville_values = [50, 5000, 50000]
pepper_names = ["Bell pepper", "Espellete pepper", "Chocolate habanero"]
pepper_descriptions = ["Not even spicy", "uncomfortable", "Practically ate pepper spray"]

pepper_list_zip = zip(scoville_values, pepper_names, pepper_descriptions)

dataframe_pepper = pd.DataFrame(pepper_list_zip)

dataframe_pepper_labeled = pd.DataFrame(pepper_list, columns = ["Scoville", "Name", "Feeling"])

# CREATING A DATAFRAME FROM DICTIONARIES

pepper_dictionary = {
    'Scoville' : [50, 5000, 500000],
    'Name' : ["Bell pepper", "Espelette pepper", "Chocolate habanero"],
    'Feeling' : ["Not even spicy", "Uncomfortable", "Practically ate pepper spray"]
}

dataframe_peppers_dict = pd.DataFrame(pepper_dictionary)

## READING A DATAFRAME FROM A FILE

# In most cases, you would be importing data/reading data instead of generating them by hand.
# Each respective filetype function follows the same syntax read_filetype(), such as read_csv(), read_excel(), read_json(), read_html(), etc.

# You can specify from which line Pandas starts reading the data, but, by default, it treats the first line of the CSV as the column name

dataframe_peppers_csv = pd.read_csv('peppers.csv')

## MANIPULATING DATAFRAMES
## ACCESSING COLUMNS

# Accessing columns is as simple as referencing it in a DataFrame, such as - dataFrameName.ColumnName or dataFrameName['ColumnName']

object_type = type(dataframe_peppers_dict['Name'])

# The underlying mechanism that makes DataFrames work are Pandas' Series objects. Each column and row is actually just a series.

dataframe_peppers_dict_truncated = dataframe_peppers_dict[['Scoville', 'Name']]

# Although DataFrame rows are just Series objects as well, the primary intention of [] notation is to select columns.
# To understand how we can access rows, we'll want to use the loc[] and iloc[] methods.

## ACCESSING/LOCATING ELEMENTS

# Pandas has two different ways of selecting individual elements - loc[] and iloc[].
# loc[] allows you to select rows and columns by using labels, like row['Value'] and column['Other Value']
# iloc[] requires that you pass in the index of the entries you want to select, so you can only use numbers

# Location by label
loc_by_label = dataframe_peppers_csv.loc[3]

# Location by Index
loc_by_index = dataframe_peppers_csv.iloc[1]

# Groups of Rows
loc_by_rows = dataframe_peppers_csv.loc[:3]

# Note that iloc[] always expects an integer. loc[] supports other dat types as well. We can use an integer here too,
# though we can use other types

# You can also access specific values of rows, instead of retrieving the entire row.

dataframe_peppers_csv_specific = dataframe_peppers_csv.loc[2, 'Scoville']

## Access column names through indexing

'Columns: ', dataframe_peppers_csv.columns
'Column at 1st index: ', dataframe.columns[2]

# MANIPULATING INDICES

# Indices are row "Labels" in a DataFrame and we can reference them when we want to select rows.
# The first way we can change the indexing of our DataFrame is by using the set-index() method.

indices = pd.Series([25, 150, 123])
dataframe_2 = dataframe.set_index(indices)

## Set Scoville as index

pepper_list_2 = [
    {'Scoville' : 50, 'Name' : "Bell pepper", 'Feeling' : "Not even spicy"},
    {'Scoville' : 5000, 'Name' : "Espelette pepper", 'Feeling' : "Uncomfortable"},
    {'Scoville' : 500000, 'Name' : "Chocolate habero", 'Feeling' : "Practically ate pepper spray"}
]

dataframe_3 = pd.DataFrame(pepper_list_2)
dataframe_3 = dataframe_3.set_index('Scoville')

# reindex() function conforms (reorders to conform) the existing DataFrame to a new set of labels.

new_index = [50, 5000, 'New value not present in the data frame']
dataframe_3 = dataframe_3.reindex(new_index)

# Since no existing row corresponds to the strings we've added, instead of having data, we have two NaN values, indicating missing values.
# You can control what value Pandas uses to fill in the missing values by setting the optional parameter fill_value:

dataframe_3 = dataframe_3.reindex(new_index, fill_value = 0)

## RESETTING INDEX

# Lets turn the index back to the default one, since we don't really want it to stay like this.

dataframe_3 = dataframe_3.reset_index(inplace= True)

## MANIPULATING ROWS

# ADDING NEW ROWS

# Adding and removing rows becomes simple if you're comfortable with using loc[].
# If you try setting a value to a row that doesn't exist, it's created, with that value.

dataframe.loc[50] =[10000, 'Serrano Pepper', 'I regret this.']

## REMOVING ROWS

# IF you want to remove a row, you pass its index to the drop() function which accepts an optional parameter, axis.
# The axis can be set to 0/index or 1/columns.

dataframe.drop(50, inplace = True)

## RENAMING ROWS

# The rename() function accepts a dictionary of changes you wish to make

dataframe.rename({0: "First", 1: "Second"}, inplace = True)

# Note that drop() and rename() both accept the optional parameter - inplace. Setting this to True (False by default) will tell Pandas-
# to change the original DataFrame instead of returning a new one. If left unset, you'll have to pack the resulting DataFrame into a new one.

## Dropping Duplicate Rows

# To drop duplicates, we can simply use the drop_duplicates() helper function, which will find all duplicate rows and drop them.

# Add two rows, with the same content

dataframe.loc[3] = [60.000, "Bird's eye chili", "4th stage of grief"]
dataframe.loc[4] = [60.000, "Bird's eye chili", "4th stage of grief"]

dataframe.drop_duplicates(inplace = True)

## REMOVING COLUMNS

# Similarly to rows, columns can be removed by calling the drop() function, the only difference being that you have to set the optional axis to 1

dataframe_peppers_dict.drop('Feeling', axis = 1, inplace = True)

## RENAMING COLUMNS

# Similarly to rows, the columns use the same rename() function. This time, we'' specifically set the columns argument,
# and add a dictionary of the old value and the new value for the column name.

dataframe_peppers_dict.rename(columns = {"Feeling": "Measure of Pain"}, inplace = True)

## DATAFRAME SHAPES

# If you'd like to check the shape of a DataFrame, you can easily access the shape property of the instance.

dataframe_peppers_dict.shape

# GROUPING DATA IN DATAFRAMES

# Grouping Data is the process of containerizing certain groups of data, based on some criteria, into categories.

students = {
    'Name' : ['John', 'John', 'Grace', 'Grace', 'Benjamin', 'Benjamin', 'Benjamin', 'Benjamin', 'John', 'Alex', 'Alex', 'Alex'],
    'Position' : [2, 1, 1, 4, 2, 4, 3, 1, 3, 2, 5, 3],
    'Year' : [2009, 2010, 2009, 2010, 2010, 2010, 2011, 2012, 2011, 2013, 2013, 2012],
    'Marks' : [408, 398, 422, 376, 401, 380, 396, 388, 356, 402, 368, 378]
}

dataframe_students = pd.DataFrame(students)

# Group students together by the Year column

group_students_year = dataframe_students.groupby('Year')

# Check the Type of the object

type(group_students_year)

# The DataFrameGroupBy object is a specific type of object, used to hold the result of the groupby() function.
# You can access different properties of this object, such as:
# groups - A dictionary of groups and their labels
# indices - A dictionary of groups and their indices

# It also offers a pretty handy method:
# get_group() - Returns a group, converted into a new DataFrame with the entries collected when grouping.

group_students_year.groups

# Dictionary of label:elements are displayed. Each group label has a list of added students.
# Take year 2010 from these and print it out.

group_students_year.get_group(2010)

# You can group data based on any column. We could've grouped by Name or Position as well.

## DESCRIPTIVE STATISTICS

## Pandas DataFrames include the describe() method, which ignores all non-numerical columns, and calculates some basic information.

students_scores_math_english = {
    'Name' : ['John', 'Alice', 'Joseph', 'Alex'],
    'English' : [64, 78, 68, 58],
    'Maths' : [76, 54, 72, 64]
}

dataframe_students_scores_math_english = pd.DataFrame(students_scores_math_english)

# Describe the statistics of the DataFrame
# Extra (Personal): Round numeral to nearest hundreths place

round(dataframe_students_scores_math_english.describe(), 2)

## DATAFRAME.HEAD() AND DATAFRAME.TAIL()

# The same as with Series objects, you can use DataFrame.head() and DataFrame.tail() to get truncated, efficient chunks of DataFrames.

dataframe_students_scores_math_english.head(3)
dataframe_students_scores_math_english.tail(3)

## RESHAPING DATAFRAMES

# 5 ADVANCED RESHAPING OPERATIONS
# Transpose, Stack, Unstack, Melt, Pivot

# TRANSPOSING A DATAFRAME

# Transposition, as the name implies, is the act of switching the places of two or more elements.
# To transpose a DataFrame means to transpose its index and columns

df_students_transposed = dataframe_students.transpose()

# alternative

df_students_transposed_alt = dataframe_students.T

# STACKING A DATAFRAME

# The stack() function reshapes the DatFrame so that the columns become parts of multi-level indices

df_students_stacked = dataframe_students.stack()

# Now, each instance, in this case - a student, haas their own Name, Year and Marks fields, instead of these columns being effectively shared.
# Each of these is a Series, and we can extract each student via an index value

student = dataframe_students.loc[1]

# You can also access the fields of specific instances.

student.Marks

## UNSTACKING A DATAFRAME

# Unstacking is the reverse process of stacking.

df_students_unstacked = df_students_stacked.unstack()

## PIVOTING A DATAFRAME

# Most of the time, we're working with narrow-form (tidy-form) data . These are also commonly known as long-form or log-data because the data -
# is written as a log of observations, one beneath the other.
# In this type, there's a column for each variable/feature, and each row is a single instance/observation

# By contrast, wide-form (short-form) data has the values of the independent variables as the row and column headings - while the values -
# of the dependent variables are contained in the cells.

new_students = {
    'Name': ['John', 'Victoria'],
    'Year': [2009,2010],
    'Marks': [408, 398]
}

df_new_students = pd.DataFrame(new_students)
df_new_students_pivoted = df_new_students.pivot(index = 'Name', columns = 'Year', values = 'Marks')

# This DataFrame is now in wide-form. Wide-form data is very commonly used for heat maps, since they are, inherently, wide-form.

## MELTING A DATAFRAME

# Melting a DataFrame is the process of reshaping it from wide-form to narrow-form.
# This is achieved by "melting" away, until there are only two columns - variable and value.

new_students_melted = df_new_students.melt()

# you can specify the id_vars and value_vars while melting the DataFrame. The column(s) passed to id_vars will be used as the new identifiers, -
# while the column(s) passed to value_vars will be used for the new values.

new_students_melted_2 = df_new_students.melt(id_vars = 'Name', value_vars = 'Marks')

## MELTING AND UN-MELTING A DATAFRAME

df_new_students
df_new_students_melted = df_new_students.melt(id_vars = 'Name', var_name = 'Variable', value_name = 'Value')
df_new_students_unmelted = df_new_students_melted.pivot(index = 'Name', columns = 'Variable')['Value'].reset_index()
df_new_students_unmelted.columns.name = None


## HOW TO ITERATE OVER ROWS

# Items()
# iterrows()
# itertuples()

# ITERATING DATAFRAMES WITH ITEMS()

dataframe_people = pd.DataFrame({
    'first_name' : ['John', 'Jane', 'Marry', 'Victoria', 'Gabriel', 'Layla'],
    'last_name' : ['Smith', 'Doe', 'Jackson', 'Smith', 'Brown', 'Martinez'],
    'age' : [34, 29, 37, 52, 26, 32]},
    index = ['id001', 'id002', 'id003', 'id004', 'id005', 'id006']
)

# We can use this to generate pairs of col_name and data. These pairs will contain a column name and every row of data for that column.

for col_name, data in dataframe_people.items():
    print("col_name: ", col_name, "\ndata: ", data)

# We can also print a particular row by passing the index number to the data as we do with Python lists.

for col_name, data in dataframe_people.items():
    print("col_name: ", col_name, "\ndata: ", data[1])

# We can also pass the index label to data

for col_name, data in dataframe_people.items():
    print("col_name: ", col_name, "\ndata: ", data['id002'])

## ITERATING DATAFRAMES WITH ITERROWS()

# While dataframe.items() iterates over the rows in column-wise, doping a cycle for each column, we can use iterrows() to get the entire row-data

for i, row in dataframe_people.iterrows():
    print(f"Index: {i}")
    print(f"{row}\n")

# Likewise, we can iterate over the rows in a certain column. Simply passing the index number or the column name to the row.

for i, row in dataframe_people.iterrows():
    print(f"Index: {i}")
    print(f"{row[0]}\n")

for i, row in dataframe_people.iterrows():
    print(f"Index: {i}")
    print(f"{row['first_name']}\n")

## ITERATING DATAFRAMES WITH ITERTUPLES()

# The itertuples() function will also return a generator, which generates row values in tuples

for row in dataframe_people.itertuples():
    print(row)

# The itertuples() method has two arguments: index and name. We can choose not to display index

for row in dataframe_people.itertuples(index = False):
    print(row)

# This generator yields namedtuples with the default name of Pandas. We can change this by passing People argument to the name parameter.

for row in dataframe_people.itertuples(index = False, name = 'People'):
    print(row)

## ITERATION PERFORMANCE WITH PANDAS

# Official Pandas documentation warns that iteration is a slow process. IF you're iterating over a DataFrame to modify the data,
# vectorization would be a quicker alternative.
# Its discouraged to modify data while iterating over rows as Pandas sometimes returns a copy of the data in the row and not its reference -
# which means that not all data will actually be changed.

## SPEED COMPARISON

# Average results in seconds:

    # Method: items(), iterrows(), itertuples()
    # Speed(s): 1.349279541666571, 3.4104003086661883, 0.412312967500279
    # Test Function: print(), print(), print()

    # Method: items(), iterrows(), itertuples()
    # Speed(s): 0.006637570998767235, 0.5749766406661365, 0.3058610513350383
    # Test Function: append(), append(), append()

## MERGING DATAFRAMES

# Merging DataFrames allows you to both create a new DataFrame without modifying the original data source or alter the original data source.
# Correlations require the use of Merging DataFrames when plotting.

# Merge DataFrames Using merge()
# Merge DataFrames Using join()
# Merge DataFrames Using append()
# Merge DataFrames Using concat()
# Merge DataFrames Using combine_first() and update()

## MERGE DATAFRAMES USING MERGE()

df1 = pd.DataFrame({
    'user_id' : ['id001', 'id002', 'id003', 'id004', 'id005', 'id006', 'id007'],
    'first_name' : ['Rivi','Wynnie','Kristos','Madalyn','Tobe','Regan','Kristin',],
    'last_name' : ['Valti', 'McMurty', 'Ivanets', 'Max', 'Riddich', 'Huyghe', 'Illis'],
    'email' : ['rvalti0@example.com', 
                'wmcmurty1@example.com',
                'kivanets2@example.com',
                'mmax3@example.com',
                'triddich4@example.com',
                'rhuyghe@example.com',
                'killis4@example.com']
})

# When designing databases, it's considered good practice to keep profile settings (like background color, avatar image link, font size etc)
# in a separate table from the user data (email, date added, etc).
# These tables can then have a one-to-one relationship

# Simulate scenario by creating df2 with image URL's and user ID's:

df2 = pd.DataFrame(
    {
        'user_id' : ['id001', 'id002', 'id003', 'id004', 'id005'],
        'image_url' : ['http://example.com/img/id001.png',
                        'http://example.com/img/id002.jpg',
                        'http://example.com/img/id003.bmp',
                        'http://example.com/img/id004.jpg',
                        'http://example.com/img/id005.png']
    }
)

# Let's combine these DataFrames with the merge() function. The merge() function accepts a lot of optional arguments, and is called -
# on the Pandas instance itself:

""" pd.merge(left, right, how = 'inner', on = None, left_on = None, right_on = None,
            left_index = False, right_index = False, sort = True,
            suffixes = ('_x', '_y'), copy = True, indicator = False,
            validate = None) """

# left and right are the two parameters that do not have optional default values - these are the names of the DataFrames that we want to merge.
# This function itself will return a new DataFrame , so it's not in-place.

df3_merged = pd.merge(df1, df2)

# Since both of our DataFrames have the column user_id with the same name, the merge() function automatically joins the two tables matching on
# that key. if we had two columns with different names, we could use left_on = 'left_column_name', and right_on = 'right_column_name'.
# When the default value of the how parameter is set to inner, a new DataFrame is generated form the intersection of the left and right DataFrames
# Therefore, if a user_id is missing in one of the tables, the row corresponding to that user_id would not be present in the merged DataFrame.

df_left_merge = pd.merge(df1, df2, how = 'left')

# With a left join, we've included all elements of the left DataFrame (df1) and every element of the right DataFrame (df2).

df_right_merge = pd.merge(df1, df2, how = 'right')

# With a right join, it would return every value from the left DataFrame that matches the right DataFrame

df_left = pd.merge(df2, df1, how = 'left', indicator = True)
df_outer = pd.merge(df2, df1, how = 'outer', indicator = True)

# The indicator flag has been set to True so that Pandas adds an additional column _merge to the end of our DataFrame.
# This column tells us if a row was found in the left, right or both DataFrames.

## MERGE DATAFRAMES USING JOIN()

# Unlike merge() which is a method of the Pandas instance, join() is a method of the DataFrame itself.

" DataFrame.join(other, on = None, how = 'left', lsuffix = '', rsuffix = '', sort = False) "

df_join = df1.join(df2, rsuffix = '_right')

# Create DataFrame with no duplicates by setting the user_id columns as an index onb both columns so it would join without a suffix

df_join_no_duplicates = df1.set_index('user_id').join(df2.set_index('user_id'))

## MERGE DATAFRAMES USING APPEND()

# concat() and append() methods return new copies of DataFrames - overusing these methods can affect the performance of your program
# Appending is very useful when you want to merge two DataFrames in row axis only.
# This means that instead of matching data on their columns, we want a new DataFrame that contains all the rows of 2 different DataFrames

# Append df2 to df1 and print the results

df_append = df1.append(df2, ignore_index = True)

# FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

# Most users choose concat() over the append() since it also provides the key matching and axis option

## MERGE DATAFRAMES USING CONCAT()

# Concatenation is a bit more flexible when compared to merge() and join() as it allows us to combine DataFrames either -
# vertically (row-wise) or horizontally (column-wise)

# The trade-off is that any data that doesn't match will be discarded. It's called on the Pandas instance, just like merge() -
# and accepts several arguments

""" pd.concat(objs, axis = 0, join = 'outer', ignore_index = False, keys = None,
            levels = None, names = None, verify_integrity = False, sort = False, copy = True) """

# Here are the most commonly used parameters for the concat() function:

# objs is the list of DataFrame objects([df1, df2, ...]) to be concatenated
# axis defines teh direction of the concatenation, 0 for the row-wise and 1 for column-wise
# join can either be inner (intersection) or outer (union)
# ignore_index by default set to False which allows the index values to remain as they were in the original DataFrames, can cause duplicate -
    # indices. If set to True, it will ignore the original values and re-assign index values in sequential order.
# Keys allows us to construct a hierarchical index. THink of it as another level of the index that appended on the outer left of the DataFrame -
    # that helps us to distinguish indices when values are not unique

# Create a new DataFrame with the same columns types with df2, but this one includes the image_url for id006 and id007:

df2_addition = pd.DataFrame(
    {
        'user_id' : ['id006', 'id007'],
        'image_url' : ['http://example.com/img/id006.png',
                        'http://example.com/img/id007.jpg']
        }
    )

df_row_concat = pd.concat([df2, df2_addition])

# 0 and 1 indices are repeating. To get entirely new and unique index values, we pass True to the ignore_index parameter

df_row_concat_new = pd.concat([df2, df2_addition], ignore_index = True)

# Concatenation can work both horizontally and vertically. To join two DataFrames together column-wise, we will need to change the axis value to 1

df_column_concat = pd.concat([df1, df_row_concat_new], axis =1)

# concat() does not do key matching like merge() or join()

## MERGE DATAFRAMES USING COMBINE_FIRST() AND UPDATE()

# In some cases, you might want to fill the missing data in your DataFrame by merging it with another DataFrame.
# By doing so, you will keep all the non-missing values in the first DataFrame while replacing all NaN values with available non-missing -
# values from the second DataFrame (if there are any).

df_first = pd.DataFrame(
    {
        'COL 1': ['X', 'X', pd.NA],
        'COL 2' : ['X', pd.NA, 'X'],
        'COL 3' : [pd.NA, 'X', 'X']
    },
    index = range(0,3)
    )

df_second = pd.DataFrame(
    {
        'COL 1' : [pd.NA, 'O', 'O'],
        'COL 2' : ['O', 'O', 'O']
    },
    index = range(0,3)
    )

# Use df_second to patch missing values in df_first:

df_tictactoe = df_first.combine_first(df_second)

# Using the combine_first() method will only replace <NA> values in the same location of another DataFrame.
# On the other hand, if we wanted to overwrite the values in df_first with the corresponding values form df_second
# (regardless they are <NA> or not), we would use the update() method

# Add another DataFrame to set of DataFrames

df_third = pd.DataFrame(
    {
        'COL 1' : ['O'],
        'COL 2' : ['O'],
        'COL 3' : ['O']
    }
)

# Update the df_first with the values from df_third

df_first.update(df_third)

# Keep in mind that unlike combine_first(), update() does not return a new DataFrame. It modifies the df_first in-place,
# altering the corresponding values.

# The overwrite parameter of the update() function is set to True by default - this is why it changes all corresponding values, -
# instead of only <NA> values. We can change it to False to replace only <NA> values

df_tictactoe.update(df_first, overwrite = False)

## HANDLING MISSING DATA

## DATA INSPECTION

file = 'out.csv'
df_out = pd.read_csv(file)

# Pandas automatically assigns NaN if the value for a particular column is an empty string '', NA or NaN.

# In our dataset, we want to consider these as missing values:

# 1. A 0 value in the Salary column
# 2. An na value in the Team column

# Most efficient way is to handle them at import-time and to achieve this we would use the na_values argument of the read_csv() method.
# This argument accepts a dictionary where the keys represent a column name and the value represents the data values that are to be considered.

df_out_na = pd.read_csv(file, na_values = {"Salary" : [0], "Team" : ['na']})

# We still have an n.a. cell in the Gender column, on index 3.
# You can map a list of values which will be treated as missing globally, in all columns.

missing_values = ['n.a', 'NA', 'n/a', 'na', 0]
df_out_na_list = pd.read_csv(file, na_values = missing_values)

## REMOVING ROWS WITH MISSING VALUES

# the dropna() function is specifically dedicated for removing all rows which contain missing values

df_out_na_list.dropna(axis=0, inplace = True)

# You can control whether you want to remove the rows containing at least 1 NaN or all NaN values by setting the how parameter in the dropna method

# how:
    # any: if any NaN values are present, drop the row
    # all: if all values are NaN, drop the row

df_out_na_list.dropna(axis = 0, inplace = True, how = 'all')

# FILLING OUT MISSING VALUES

# Fill NaN's with Mean, Median or Mode of the data
# Fill NaN's with a constant value
# Forward Fill or Backward Fill NaN's
# Interpolate Data and Fill NaN's

## FILL MISSING DATAFRAME VALUES WITH COLUMN MEAN, MEDIAN AND MODE

df_out_na_list['Salary'].fillna(df_out_na_list['Salary'].median(), inplace = True)
df_out_na_list['Salary'].fillna(int(df_out_na_list['Salary'].mean()), inplace = True)
df_out_na_list['Salary'].fillna(int(df_out_na_list['Salary'].mode()), inplace = True)

## FILL MISSING DATAFRAME VALUES WITH A CONSTANT

# You could also decide to fill the NaN-marked values with a constant value.

df_out_na_list['Salary'].fillna(0, inplace = True)

# Square one was having unstandardized missing values, and this functionally equivalent to having NaNs.
# The difference is, you can use values like 0 and perform operations on them, that you can't perform on NaN.

## FORWARD FILL MISSING DATAFRAME VALUES

# This method would fill the missing values with first non-missing value that occurs before it.

df_out_na_list['Salary'].fillna(method = 'ffill', inplace = True)

## BACKWARD FILL MISSING DATAFRAME VALUES

# This method would fill the missing values with first non-missing value that occurs after it

df_out_na_list['Salary'].fillna(method = 'bfill', inplace = True)

## FILL MISSING DATAFRAME VALUES WITH INTERPOLATION

# This method uses mathematical interpolation to determine what value would have been in the place of a missing value.
# The interpolate() function can be used to achieve this, and for the polynomial and spline methods, you'll also have to specify
# the order of that method.

df_out_na_list['Salary'].interpolate(method = 'polynomial', order = 5, inplace = True)
df_out_na_list['Salary'].interpolate(method = 'spline', order = 5, inplace = True)
df_out_na_list['Salary'].interpolate(method = 'linear', inplace = True)

## READING AND WRITING CSV FILES

## WHAT IS A CSV FILE?

# Nothing more than a simple text file, following a few formatting conventions. However, it is the most common, simple and easiest
# method to store tabular data.
# This format arranges tables by following a specific structure divided into rows and columns. It is these rows and columns that contain your data
# A new line terminates each row to start the next row.
# Similarly, a delimiter, usually a comma, separates columns within each row.

## READING CSV FILES WITH READ_CSV()

titanic_csv = 'titanic.csv'
titanic_data = pd.read_csv(titanic_csv)

titanic_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_data_url = pd.read_csv(titanic_url)

## CUSTOMIZING HEADERS

# You can set headers either after reading the file, simply by assigning the columns field of the DataFrame instance another list
# or while reading the CSV in the first place

col_names = [
    'Id',
    'Survived',
    'Passenger Class',
    'Full Name',
    'Gender',
    'Age',
    'SibSp',
    'Parch',
    'Ticket Number',
    'Price', 'Cabin',
    'Station'
]

titanic_data = pd.read_csv(titanic_url, names = col_names)

## SKIPPING ROWS WHILE READING CSV

titanic_data = pd.read_csv(titanic_url, names = col_names, skiprows = [0])

## REMOVING HEADERS

titanic_data = pd.read_csv(titanic_url, header = None, skiprows = [0])

## SPECIFYING DELIMITERS

# You'll eventually probably encounter a CSV file that doesn't actually use commas to separate data.
# In such cases, you can use the sep argument to specify other delimiters.

" titanic_data = pd.read_csv(titanic_url, sep = ';') "

## WRITING CSV FILES WITH TO_CSV()

# Call the write_csv() function on the DataFrame instance to turn DataFrames into CSV files.
# When writing a DataFrame to a CSV file, you can also change the column names, using the columns argument, or specify a delimiter via the sep arg
# If you don't specify either of these, you'll end up with a standard Comma-Separated Value file

cities = pd.DataFrame([['Sacramento', 'California'], ['Miami', 'Florida']], columns = ['City', 'State'])
cities.to_csv('cities.csv')

# Poor Formatting - We've still got the indices from the DataFrame, which also puts a weird missing spot before the column names.

df_poor_format = pd.read_csv('cities.csv')

# The indices from the DataFrame ended up becoming a new column, which is now Unnamed.
# When saving the file, make sure to drop the index of the DataFrame, since it's not really a part of the data - it's a part of the DataFrame

cities.to_csv('cities.csv', index = False)

# Properly Constructed

df_proper_format = pd.read_csv('cities.csv')

## CUSTOMIZING HEADERS

new_column_names = ['City_Name', 'State_Name']
cities.to_csv('cities.csv', index = False, header = new_column_names)

df_new_header = pd.read_csv('cities.csv')

## CUSTOMIZING DELIMITER

# ADJUST DELIMITER TO A NEW ONE

cities.to_csv('cities.csv', index = False, sep = ';')

## HANDLING MISSING VALUES

# Sometimes DataFrames have missing values that we've left as NaN or NA.
# You can use the na_rep argument and set the value to be put instead of a missing value

new_cities = pd.DataFrame([['Sacramento', 'California'], ['Miami', 'Florida'], ['Washington DC', pd.NA]], columns = ['City', 'State'])
new_cities.to_csv('new_cities.csv', index = False, na_rep = 'Unknown')