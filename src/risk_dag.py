import time
from itertools import product
from fractions import Fraction
from collections import deque

# Count W (both defenders die), L (both attackers die), and T (one loss each)
def probability_space():
	start_time = time.time()
	all_rolls = list(product(range(1, 7), repeat=5))

	# Initialize counters
	W = 0  # Both defenders die
	L = 0  # Both attackers die
	T = 0  # One attacker, one defender lost
	N = len(all_rolls)
	assert N == 7776  # 6^5 = 7776

	# Iterate through all possible dice rolls
	for roll in all_rolls:
		att_rolls = sorted(roll[:3], reverse=True)  # First 3 elements of a 5 element set
		def_rolls = sorted(roll[3:], reverse=True)  # Last 2 elements of a 5 element set

		# Compare dice outcomes
		attacker_losses = 0
		defender_losses = 0

		# First dice pair
		if att_rolls[0] > def_rolls[0]:
			defender_losses += 1
		else:  # Defender wins ties
			attacker_losses += 1

		# Second dice pair
		if att_rolls[1] > def_rolls[1]:
			defender_losses += 1
		else:  # Defender wins ties
			attacker_losses += 1

		# Categorize outcome
		if defender_losses == 2:
			W += 1  # Both defenders die
		elif attacker_losses == 2:
			L += 1  # Both attackers die
		else:
			T += 1  # One attacker, one defender lost

	assert W + L + T == N  # Sanity check
	print(f"probability_space execution time: {time.time() - start_time:.4f} seconds")
	return W, L, T, N

# Build the graph that describes starting from the start of the battle to when a dice is removed: either
# - the attackers being reduced to 2 attackers
# - the defender being reduced to 1 defender
def construct_DAG(attackers, defenders, W, L, T):
	start_time = time.time()
	DAG = {}
	queue = deque([(attackers, defenders)])
	visited = set()
	
	while queue:
		A, D = queue.popleft()
		if A <= 2 or D <= 1:
			continue
		
		W_next = (A, D - 2)
		L_next = (A - 2, D)
		T_next = (A - 1, D - 1)
		
		DAG[(A, D)] = {W_next: W, L_next: L, T_next: T}
		
		for next_state in [W_next, L_next, T_next]:
			if next_state not in DAG and next_state not in visited:
				visited.add(next_state)
				queue.append(next_state)
	
	print(f"construct_DAG execution time: {time.time() - start_time:.4f} seconds")
	return DAG

# Multiply probabilities of each branch from the starting node and sum all paths together that result in A < 3
# The DAG has 4 absorbing states:
# (2,2) – Attackers drop to 2, can't roll 3 dice.
# (2,1) – Attackers drop to 2, defenders also incidentally drop to 1.
# (N,1) for N ≥ 3 – Any scenario where the defender has only 1 unit left, meaning further attacks will always result in a 1v1 dice roll.
# (N,0) for N ≥ 3 – Defender is wiped out, attackers win.
def traverse_DAG(DAG, start):
	start_time = time.time()
	queue = [(start, Fraction(1))]
	probability_sum = Fraction(0)
	
	while queue:
		(A, D), prob = queue.pop(0)
		
		# If we reach (2,2) or (2,1), add to probability sum
		if (A, D) == (2,2) or (A, D) == (2,1):
			probability_sum += prob
			continue
		
		# Continue traversing if the state has transitions
		if (A, D) in DAG:
			for next_state, transition_prob in DAG[(A, D)].items():
				queue.append((next_state, prob * transition_prob))
	
	print(f"traverse_DAG({start}) execution time: {time.time() - start_time:.4f} seconds")
	return probability_sum

# Call the function and unpack the values
W, L, T, N = probability_space()
DAG = construct_DAG(75, 10, Fraction(W, N), Fraction(L, N), Fraction(T, N))
probability_map = {i: traverse_DAG(DAG, (i, 10)) for i in range(50, 76)}

scales = [(1e15, "100 trillion"), (1e14, "10 trillion"), (1e13, "trillion"),
		  (1e12, "100 billion"), (1e10, "10 billion"), (1e9, "billion"),
		  (1e8, "100 million"), (1e7, "10 million"), (1e6, "million"),
		  (1e5, "100 thousand"), (1e4, "10 thousand"), (1e3, "thousand"),
		  (1e2, "hundred")]

for i in range(50, 76):
	p = float(probability_map[i])
	order = next((label for value, label in scales if p < 1/value), "less than hundred")
	print(f"p({i} lost): {p:.2e}, worse than 1 in {order}", end='  |  ' if (i + 1) % 5 else '\r\n')
