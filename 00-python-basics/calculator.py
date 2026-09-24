try:
    result = eval(input("Enter equation (e.g. 2 + 3 * 4): "))
    print("Result:", result)
except Exception as e:
    print("Invalid equation:", e)
