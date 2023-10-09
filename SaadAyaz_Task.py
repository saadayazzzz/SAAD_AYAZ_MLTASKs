def total_permutations(num):
    if len(num) == 0:
        return []

    def backtrack(start, end):
        if start == end:
            results.append(num[:])
        else:
            for x in range(start, end):
                num[start], num[x] = num[x], num[start]
                backtrack(start + 1, end)
                num[start], num[x] = num[x], num[start]  # backtrack

    results = []
    backtrack(0, len(num))
    return results

# Get input array from the user
array = input("Enter the elements of array: ")
array = [int(num) for num in array.split(",")]

# Find permutations
permutations = total_permutations(array)
print("The total permutations are:", permutations)