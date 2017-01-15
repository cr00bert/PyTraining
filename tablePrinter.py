tableData = [['apples', 'oranges', 'cherries', 'banana'],
             ['Alice', 'Bob', 'Carol', 'David'],
             ['dogs', 'cats', 'moose', 'goose']]


def printTable(table):
    length = 0
    xLength = len(table)
    yLength = len(table[0])

    for y in range(yLength):
        temp = 0
        for x in range(xLength):
            temp += len(table[x][y])
        if temp > length:
            length = temp

    for y in range(yLength):
        text = []
        for x in range(xLength):
            text.append(table[x][y])
        print(' '.join(text).rjust(length + xLength - 1))


printTable(tableData)

