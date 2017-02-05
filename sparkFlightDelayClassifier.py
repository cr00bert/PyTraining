import sparkSetup as ss

import urllib.request as url
#f = url.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "Data/kddcup.data.gz")

dataFile = "C:/Users/CR/PycharmProjects/PyTraining/Data/kddcup.data.gz"

raw_data = ss.sqlContext.read(dataFile)

print("Train data size is " + format(raw_data.count()))