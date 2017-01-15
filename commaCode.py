def commaFunc(list):
    result = ''
    for i in range(0, len(list)-1):
        result = result + list[i] + ', '
    result = result + 'and ' + list[len(list)-1]
    return result


spam = ['apples', 'bananas', 'tofu', 'cats']


print(commaFunc(spam))
