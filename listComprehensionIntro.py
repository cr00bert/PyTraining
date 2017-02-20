# List of strings
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# List comprehension
fellow1 = [member for member in fellowship if len(member) >= 7]

# Generator expression
fellow2 = (member for member in fellowship if len(member) >= 7)

print('\nPrinting list:')
print(fellow1)

print('\nPrinting generator values using next():')
print(next(fellow2))
print(next(fellow2))
print('Remainder of generator iterables:')
for val in fellow2:
    print(val)
