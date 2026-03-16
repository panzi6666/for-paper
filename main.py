from itertools import combinations

letters = 'abcdefghijklmnopqrstuvwxyz'

for r in range(1, len(letters)+1):
    for combo in combinations(letters, r):
        print(''.join(combo))