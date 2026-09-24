nums = [int(x) for x in input("Enter numbers separated by spaces: ").split()]
evens = [x for x in nums if x % 2 == 0]

if evens:
    print("Even numbers:", *evens)
else:
    print("No even numbers found.")
