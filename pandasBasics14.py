####################
# Basic Formats
####################

# Old method
print('%s %s' % ('one', 'two'))
print('%d %d' % (1, 2))
# New method
print('{} {}'.format('one', 'two'))
print('{} {}'.format(1, 2))

# Positional numbers are advised to use
print('{1} {0}'.format(1, 2))

######################
# Padding and aligning
######################

# Align right
# Old method
print('%10s' % ('test',))
# New method
print('{:>10}'.format('test'))
print('{:_>10}'.format('test'))

# Alight left
# Old method
print('%-10s' % ('test',))
# New method
print('{:10}'.format('test'))
print('{:_<10}'.format('test'))

# Center align
print('{:_^10}'.format('test'))

########################
# Long string truncation
########################

# Old method
print('%.5s' % ('xylophone',))
# New method
print('{:.5}'.format('xylophone'))

# Old method
print('%-10.5s' % ('xylophone',))
# New method
print('{:_<10.5}'.format('xylophone'))

#################
# Padding numbers
#################

# Old method
print('%4d' % (42,))
# New method
print('{:4d}'.format(42))

