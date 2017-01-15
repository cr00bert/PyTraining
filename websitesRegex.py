#TODO Find website URLs that begin with http:// or https://.

import pyperclip, re
# Copy text from clipboard
text = pyperclip.paste()

#Todo Define regex

websiteRegex = re.compile(r'''(
(http://|https://)?
(www\.)?
([a-zA-Z0-9.-])+
(\.[a-zA-Z]{2,4})+
)''', re.VERBOSE)

#text = 'http://youtube.com'
#Todo search text

matches = []
for groups in websiteRegex.findall(text):
    matches.append(groups[0])

if len(matches) > 0:
    pyperclip.copy('\n'.join(matches))
    print('Copied to clipboard.')
    print('\n'.join(matches))
else:
    print('No matches found.')

#todo Paste back to clipboard