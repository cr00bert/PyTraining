import re


def regexStrip(string, char = None):
    if char == None:
        char = r'\s'

    regB = re.compile(r'^'+char+r'*')
    regE = re.compile(char + r'*$')

    stripB = re.sub(regB, '', string)
    stripE = re.sub(regE, '', stripB)

    return(stripE)


string = "@@@     Get rid of the ws. "

print(regexStrip(string, '@'))