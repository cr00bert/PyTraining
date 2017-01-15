def collatz(number):
    if number % 2 == 0:
        result = number // 2
        print(str(result))
        return result
    else:
        result = 3 * number + 1
        print(str(result))
        return result

print('Enter number:')
try:
    userInput = int(input())
except ValueError:
    print('Please enter an integer.')

while True:
    userInput = collatz(userInput)
    if userInput == 1:
        break

