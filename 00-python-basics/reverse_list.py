def rev_list(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    arr[-1::]
    return arr


data = [int(x) for x in input().split()]
rev_list(data)
print(*data)
