import re


uppCa = re.compile(r'''[A-Z]+''', re.VERBOSE)
lowCa = re.compile(r'''[a-z]+''', re.VERBOSE)
matchesU = []
matchesL = []

while True:
    print("Type in your new password:")
    pw = input()
    if len(pw) < 8:
        print("Your password does not meet the minimum length criteria.")
        continue
    if pw.isalpha():
        print("Your password must contain at least one digit.")
        continue
    matchesU = re.findall(uppCa, pw)
    matchesL = re.findall(lowCa, pw)
    if len(matchesL) + len(matchesU) > 0:
        if len(matchesU) == 0:
            print("Your password must contain at least one uppercase letter.")
            continue
        if len(matchesL) == 0:
            print("Your password must contain at least one lowercase letter.")
            continue
    else:
        print("Your password must contain at one letter.")
        continue
    print("Your password has been saved.")
    break

