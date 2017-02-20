# Process for loading first 1000 lines of the file
# Open a connection to the file
with open('Data\\WDI\\WDI_Data.csv') as file:

    # Skip column names
    file.readline()

    # Initialize and empty dictionary
    counts_dict = {}

    # Process the first 1000 rows of the file
    for j in range(0, 1000):

        # Split the current line into a list
        line = file.readline().split(',')

        # Get the value for the first column:
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add the value to the dictionary:
        else:
            counts_dict[first_col] = 1

print(counts_dict)

# Lazy evaluator of file
def read_large_file(file_object):
    """A generator functions to read a large file lazily."""

    # Loop until the end of file
    while True:

        # Read line from the file
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data

# Open a  connection to the file
with open('Data\\WDI\\WDI_Data.csv') as file:

    # Create a generator object for the file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))


# Use the generator function to read the whole file and create dictionary
counts_dict = {}

# Open the connection to the file
with open('Data\\WDI\\WDI_Data.csv') as file:

    # Iteratore over the generator from the the function
    for line in read_large_file(file):
        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print result
print(counts_dict)