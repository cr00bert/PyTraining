#TODO Clean up dates in different date formats (such as 3/14/2015, 03-14-2015, and 2015/3/14) by replacing them with dates in a single, standard format.


import pyperclip, re

text = pyperclip.paste()

#Todo Define regex

dateRegex = re.compile(r'''(
([0-1]{1}\d|[1-9]|\d{4})
(/|.|-)?
([0-1]{1}\d|[1-9])
(/|.|-)?
(\d{4}|\d{2})
)''', re.VERBOSE)

#text = '3/14/2015  03/14/2017 03.14.2017 2015/3/14  12-14-2013'

#Todo search text
matches = []
for groups in dateRegex.findall(text):
    if len(groups[5]) == 4:
        matches.append('/'.join([groups[5], ('0'+groups[1])[-2:], ('0'+groups[3])[-2:]]))
    else:
        matches.append('/'.join([groups[1], ('0'+groups[3])[-2:],('0'+groups[5])[-2:]]))

if len(matches) > 0:
    pyperclip.copy('\n'.join(matches))
    print('Copied to clipboard.')
    print('\n'.join(matches))
else:
    print('No dates were found.')

#todo Paste back to clipboard