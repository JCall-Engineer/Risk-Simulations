from itertools import product
from fractions import Fraction
from math import factorial

# Count W (both defenders die), L (both attackers die), and T (one each dies)
def probability_space():
	all_rolls = list(product(range(1, 7), repeat=5))

	# Initialize counters
	W = 0  # Both defenders die
	L = 0  # Both attackers die
	T = 0  # One attacker and one defender each dies
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
			T += 1  # One attacker, one defender, each dies

	assert W + L + T == N  # Sanity check
	return Fraction(W, N), Fraction(L, N), Fraction(T, N)

# Compute the probability of transitioning from (A1, D1) to (A2, D2) using combinatorial calculations
def compute_probability(start, end, W, L, T):
	A1, D1 = start
	A2, D2 = end
	
	deltaA = A1 - A2
	deltaD = D1 - D2
	
	# It is impossible for an odd number of troops to be lost in any given battle
	if ((deltaA + deltaD) % 2 > 0):
		return 0

	W_count = deltaD // 2  # Number of times W happened
	L_count = deltaA // 2  # Number of times L happened

	# Minimum number of T transitions needed to balance parity
	T_min = deltaA % 2

	# Maximum possible T transitions limited by available attackers/defenders
	T_max = min(L_count, W_count) + T_min

	#print(f"DEBUG: start={start}, end={end}, deltaA={deltaA}, deltaD={deltaD}, T_min={T_min}, T_max={T_max}")
	
	total_prob = 0
	for T_count in range(T_min, T_max + 1):
		# To determine every path possible make sure we account for every possible arrangement of W, L, and T
		# For T > T_min, every added T replaces one W and one L
		adjusted_W = W_count - (T_count - T_min)
		adjusted_L = L_count - (T_count - T_min)
		
		assert adjusted_W >= 0 and adjusted_L >= 0, f"Invalid W/L counts: start={start} end={end} W={adjusted_W}, L={adjusted_L}, T={T_count}"
		
		# The multinomial coefficient counts the number of ways to arrange W, L, and T transitions in a sequence
		# It is used instead of a simple permutation formula (nPm) because we are arranging repeated elements
		total_transitions = adjusted_W + adjusted_L + T_count
		permutations = (factorial(total_transitions) // 
						(factorial(adjusted_W) * factorial(adjusted_L) * factorial(T_count)))
		
		# All permutations of this path have the same probability
		p_path = (W ** adjusted_W) * (L ** adjusted_L) * (T ** T_count)
		total_prob += permutations * p_path
	
	return total_prob

# Sum multiple paths in the hypothetical DAG together for a composite probability
def WLT_union(start, ends, W, L, T):
	total_prob = 0
	for end in ends:
		total_prob += compute_probability(start, end, W, L, T)
	return total_prob

# Compute probabilities for the range [50, 75] to reach (2, n)
# Why? if we consider attackers dropping to 2 as a loss then the error term for rolling fewer than 5 dice is not significant at this scale
W, L, T = probability_space()
probability_map = {i: WLT_union((i, 10), [(2, n) for n in range(1, 11)], W, L, T) for i in range(50, 76)}

scales = [(1e15, "100 trillion"), (1e14, "10 trillion"), (1e13, "a trillion"),
		  (1e12, "100 billion"), (1e10, "10 billion"), (1e9, "a billion"),
		  (1e8, "100 million"), (1e7, "10 million"), (1e6, "a million"),
		  (1e5, "100 thousand"), (1e4, "10 thousand"), (1e3, "a thousand"),
		  (1e2, "a hundred")]

for i in range(50, 76):
	p = float(probability_map[i])
	order = next((label for value, label in scales if p < 1/value), "<100")
	print(f"p({i} lost): {p:.2e}, worse than 1 in {order}")
