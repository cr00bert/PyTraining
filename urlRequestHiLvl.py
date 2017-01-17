# Import package
import requests

# Specify the url: url
url = 'http://docs.datacamp.com/teach/'

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
text = r.text

# Print the html
print(text)
