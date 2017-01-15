stuff = {'gold coin': 42, 'rope': 1}


def displayInventory(inv):
    print('Inventory:')
    totalCount = 0
    for k, v in inv.items():
        print(v, end=' ')
        print(str(k))
        totalCount += v
    print('Total number of items: ' + str(totalCount) +'\n')


def addToInventory(inv, newItems):
    for item in newItems:
        if inv.get(item, 0) == 0:
            inv.setdefault(item, 1)
        else:
            inv[item] += 1
    return inv


displayInventory(stuff)
dragonLoot = ['gold coin', 'dagger', 'gold coin', 'gold coin', 'ruby']
stuff = addToInventory(stuff, dragonLoot)
displayInventory(stuff)
