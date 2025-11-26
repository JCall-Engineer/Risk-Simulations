from __future__ import annotations
from itertools import product
from fractions import Fraction
from math import factorial
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ProbabilitySpace:
	name: Node
	dice: int
	W: int
	T: int
	L: int
	N: int
	P_W: Fraction
	P_T: Fraction
	P_L: Fraction

	def __iter__(self):
		"""Allows unpacking as a tuple: W, T, L = space (primarily used in tests)"""
		return iter((self.P_W, self.P_T, self.P_L))

class Node:
	attackers: int
	defenders: int

	# Cast from tuple
	def __init__(self, *args, **kwargs):
		if len(args) == 1 and isinstance(args[0], tuple):
			self.attackers, self.defenders = args[0]
		elif len(args) == 2:
			self.attackers, self.defenders = args
		else:
			self.attackers = kwargs.get('attackers', 0)
			self.defenders = kwargs.get('defenders', 0)

	def outcomes(self):
		"""Yield (outcome_node, probability) for all possible battle outcomes"""
		match min(2, self.attackers, self.defenders): # At most 2 dice can be compared
			case 2:
				# 3v2, or 2v2: 2 dice - W, T, L possible
				yield (Node(self.attackers, self.defenders - 2), self.space.P_W)
				if self.space.T > 0:
					yield (Node(self.attackers - 1, self.defenders - 1), self.space.P_T)
				yield (Node(self.attackers - 2, self.defenders), self.space.P_L)
			case 1:
				# 1vn or nv1: 1 dice - only W and L possible
				yield (Node(self.attackers, self.defenders - 1), self.space.P_W)
				yield (Node(self.attackers - 1, self.defenders), self.space.P_L)
			case 0:
				raise ValueError(f"Cannot compute outcomes for {self}: no dice to roll")

	def is_valid(self):
		"""
		Check if this node represents a valid game state.

		A node is invalid if:
		- Either attackers or defenders is negative
		- Both attackers and defenders are zero (no Risk battle can result in mutual annihilation)
		"""
		return not any(i < 0 for i in [self.attackers, self.defenders]) and self.attackers + self.defenders > 0

	def has_edges(self):
		"""
		Check if this node has outgoing edges in the battle DAG.

		Returns True only when both attackers and defenders are present, meaning
		a battle can occur and transitions to other nodes are possible. Terminal
		states (where one side has 0 troops) have no outgoing edges.
		"""
		return self.attackers > 0 and self.defenders > 0

	@property
	def space(self) -> ProbabilitySpace:
		return probability_space(self)

	def __hash__(self):
		"""Makes Node hashable so it can be used as a dict key or in sets"""
		return hash((self.attackers, self.defenders))

	def __eq__(self, other):
		if isinstance(other, Node):
			return self.attackers == other.attackers and self.defenders == other.defenders
		if isinstance(other, tuple):
			if len(other) != 2: return False
			return self.attackers == other[0] and self.defenders == other[1]
		return False

	def __sub__(self, other):
		"""Allows a handy delta = start - end"""
		if isinstance(other, Node):
			return Node(
				self.attackers - other.attackers,
				self.defenders - other.defenders
			)
		return NotImplemented

	def __iter__(self):
		"""Allows unpacking as a tuple: a, d = node"""
		return iter((self.attackers, self.defenders))

	def __repr__(self):
		"""Prints the same as a tuple: (a, d)"""
		return f"({self.attackers}, {self.defenders})"

computed_spaces: dict[Node, ProbabilitySpace] = {}
def probability_space(attackers: int | tuple[int, int] | Node = 3, defenders: int = 2) -> ProbabilitySpace:
	"""
	Count all possible dice rolls and categorize them into Win, Tie, and Loss outcomes.

	Computes the probability space for a given attacker/defender configuration by enumerating
	all possible dice rolls and comparing outcomes according to Risk rules. Results are cached
	since this function is called frequently during probability calculations.

	Args:
		attackers: Number of attacking dice (1-3), or a Node/tuple of (attackers, defenders)
		defenders: Number of defending dice (1-2), ignored if attackers is a Node/tuple

	ProbabilitySpace containing:
		- name: The Node representing this configuration
		- dice: Total number of dice rolled
		- W, T, L: Count of wins, ties, losses
		- N: Total possible outcomes
		- P_W, P_T, P_L: Probabilities as Fractions
	"""
	if isinstance(attackers, (Node, tuple)):
		attackers, defenders = attackers

	attackers = min(3, attackers)
	defenders = min(2, defenders)
	dice = attackers + defenders

	assert attackers > 0
	assert defenders > 0
	assert dice >= 2

	index = Node(attackers, defenders)
	if index in computed_spaces:
		return computed_spaces[index]

	all_rolls = list(product(range(1, 7), repeat=dice))

	# Initialize counters
	W = 0  # Both defenders die
	L = 0  # Both attackers die
	T = 0  # One attacker and one defender each dies
	N = len(all_rolls)

	power_6 = [1, 6, 36, 216, 1296, 7776]
	assert N == power_6[dice]

	# Iterate through all possible dice rolls
	for roll in all_rolls:
		att_rolls = sorted(roll[:attackers], reverse=True)
		def_rolls = sorted(roll[attackers:], reverse=True)
		assert len(att_rolls) == attackers
		assert len(def_rolls) == defenders

		# Compare dice outcomes
		compare = min(attackers, defenders)

		attacker_losses = 0
		defender_losses = 0

		for i in range(0, compare):
			if att_rolls[i] > def_rolls[i]:
				defender_losses += 1
			else:
				attacker_losses += 1

		# Categorize outcome (handles 1 units lost or 2 units lost)
		if defender_losses > attacker_losses:
			W += 1
		elif attacker_losses > defender_losses:
			L += 1
		else:
			T += 1

	assert W + L + T == N  # Sanity check
	result = ProbabilitySpace(
		name=index,
		dice=dice,
		W = W,
		T = T,
		L = L,
		N = N,
		P_W = Fraction(W, N),
		P_T = Fraction(T, N),
		P_L = Fraction(L, N),
	)
	computed_spaces[index] = result
	return result

# Precompute the probability spaces
for a in range(3):
	for d in range(2):
		probability_space(a + 1, d + 1)

def constant_space_probability(start: Node, end: Node) -> Fraction:
	"""
	Compute the probability of transitioning from start to end using combinatorial calculations.

	This function is optimized specifically for the 3v2 probability space where the combinatorial
	savings are significant. Technically it could work for 2v2: the logic assumes every combat
	results in exactly 2 troops lost (W_max, W_edges, L_max, L_edges all use //2). Modifying this
	for other spaces provides negligible benefit compared to handling them uniformly with dynamic
	programming. Let this function do what it's good at: 3v2, and nothing else.
	"""
	if (any(i < 0 for i in [start.attackers, start.defenders, end.attackers, end.defenders])):
		raise ValueError(f"Negative troop counts are invalid: start={start}, end={end}")

	if (any(i == 0 for i in [start.attackers, start.defenders])):
		raise ValueError(f"There are no dice to be rolled at {start}")
	if (all(i == 0 for i in [end.attackers, end.defenders])):
		raise ValueError(f"{end} is not a state that can be reached")

	space = computed_spaces[Node(3, 2)]
	if start.space != space:
		raise ValueError(f"This optimization only applies to 3v2, not {start.space.name}")

	if end.space != space:
		raise ValueError(f"The number of dice used changes between {start} and {end}, this function is unable to compute probability over a non-uniform probability space")

	delta = start - end
	if (delta.attackers < 0 or delta.defenders < 0):
		return Fraction(0) # It is impossible for one size to gain troops in combat
	if (all(i >= 2 for i in [start.attackers, start.defenders]) and ((delta.attackers + delta.defenders) % 2 > 0)):
		return Fraction(0) # It is impossible for an odd number of troops to be lost in a battle with 2+ attackers and defenders

	W_max = delta.defenders // 2  # Number of times W happened
	L_max = delta.attackers // 2  # Number of times L happened

	# Minimum number of T transitions needed to balance parity
	T_min = delta.attackers % 2

	# Maximum possible T transitions: each W/L pair can be replaced by 2 T's
	T_max = 2 * min(L_max, W_max) + T_min

	total_probability = Fraction(0)
	for T_edges in range(T_min, T_max + 1, 2):
		# To determine every path possible make sure we account for every possible arrangement of W, L, and T
		# One W and one L are replaced for every 2 T transitions added beyond T_min
		W_edges = W_max - ((T_edges - T_min) // 2)
		L_edges = L_max - ((T_edges - T_min) // 2)

		assert W_edges >= 0 and L_edges >= 0, f"Invalid W/L counts: start={start} end={end} W={W_edges}, L={L_edges}, T={T_edges}"

		# The multinomial coefficient counts the number of ways to arrange W, L, and T transitions in a sequence
		# It is used instead of a simple permutation formula (nPm) because we are arranging repeated elements
		total_edges = W_edges + L_edges + T_edges
		multinomial = (factorial(total_edges) // 
						(factorial(W_edges) * factorial(L_edges) * factorial(T_edges)))

		# All permutations of this path have the same probability
		path_probability = (space.P_W ** W_edges) * (space.P_L ** L_edges) * (space.P_T ** T_edges)
		total_probability += multinomial * path_probability

	return total_probability

def compute_probability(start: Node, end: Node) -> Fraction:
	if not start.is_valid() or not start.has_edges() or not end.is_valid():
		return Fraction(0)

	if start.attackers < end.attackers or start.defenders < end.defenders:
		return Fraction(0)

	if start == end:
		return Fraction(1)

	# Handle the simple case early: both nodes in 3v2 space
	if end.has_edges() and start.space == end.space and start.space.name == Node(3, 2):
		return constant_space_probability(start, end)

	# Complex case: track probability using dynamic programming
	reach_probability: dict[Node, Fraction] = defaultdict(lambda: Fraction(0))

	# We can shortcut across the 3v2 boundary (the largest space)
	# This lists all nodes between start and end that have at least one edge that leaves 3v2
	boundaries_3v2 = list(filter(lambda node: node.attackers <= start.attackers and node.defenders <= start.defenders, [
		# See src/visualization.py or out/visualization.png to aid in understanding the logic
		Node(3, 2), # W -> victory, T -> 2v1, L -> 1v2
		*(Node(3, d) for d in range(3, start.defenders + 1)),  # Right edge: L -> 1v2
		*(Node(a, 2) for a in range(4, start.attackers + 1)),  # Bottom edge: W -> victory
		Node(4, 3), # W -> 3v1, T -> 3v2 (later ignored), L -> 2v2
		*(Node(4, d) for d in range(4, start.defenders + 1)),  # One in from right edge: L -> 2v2
		*(Node(a, 3) for a in range(5, start.attackers + 1)),  # One in from bottom edge: W -> 3v1
	]))
	for boundary in boundaries_3v2:
		reach_probability[boundary] = constant_space_probability(start, boundary)
	if len(boundaries_3v2) == 0:
		reach_probability[start] = Fraction(1)

	# Process in topological order (high troops → low troops)
	for total in range(start.attackers + start.defenders, end.attackers + end.defenders - 1, -1):
		for a in range(end.attackers, start.attackers + 1):
			d = total - a
			node = Node(a, d)

			if d < end.defenders or d > start.defenders:
				continue

			if node not in reach_probability or reach_probability[node] == 0:
				continue

			if not node.has_edges():
				continue

			for outcome, edge_prob in node.outcomes():
				assert outcome.is_valid()

				# Ensure we don't double count 3v2 optimizations
				if outcome not in boundaries_3v2:
					reach_probability[outcome] += reach_probability[node] * edge_prob

	return reach_probability[end]

def paths_union(start: Node, ends: list[Node]):
	"""
	Sum probabilities of multiple paths in the DAG to compute composite probability.

	Used to calculate the probability of reaching any state in a set of end states.
	For example, computing the probability of victory (reaching any state with 0 defenders)
	or the probability of losing exactly N or more attackers.
	"""
	total_prob = Fraction(0)
	for end in ends:
		total_prob += compute_probability(start, end)
	return total_prob

import unittest
class TestProbabilities(unittest.TestCase):
	def test_crossing_space(self):
		W_3v2, T_3v2, L_3v2 = probability_space(Node(3, 2))
		W_3v1, T_3v1, L_3v1 = probability_space(Node(3, 1))
		W_2v2, T_2v2, L_2v2 = probability_space(Node(2, 2))
		W_2v1, T_2v1, L_2v1 = probability_space(Node(2, 1))
		W_1v2, T_1v2, L_1v2 = probability_space(Node(1, 2))
		W_1v1, T_1v1, L_1v1 = probability_space(Node(1, 1))

		# No ties with one dice
		self.assertEqual(T_3v1, Fraction(0))
		self.assertEqual(T_2v1, Fraction(0))
		self.assertEqual(T_1v1, Fraction(0))
		self.assertEqual(T_1v2, Fraction(0))

		# 3v3 transitions
		p = compute_probability(Node(75, 2), Node(75, 0))
		self.assertEqual(p, W_3v2)

		p = compute_probability(Node(3, 2), Node(2, 1))
		self.assertEqual(p, T_3v2)

		p = compute_probability(Node(4, 10), Node(2, 10))
		self.assertEqual(p, L_3v2)

		# 3v1 transitions
		p = compute_probability(Node(75, 1), Node(75, 0))
		self.assertEqual(p, W_3v1)

		p = compute_probability(Node(3, 1), Node(2, 1))
		self.assertEqual(p, L_3v1)

		# 2v2 transitions
		p = compute_probability(Node(2, 2), Node(2, 0))
		self.assertEqual(p, W_2v2)

		p = compute_probability(Node(2, 2), Node(1, 1))
		self.assertEqual(p, T_2v2)

		p = compute_probability(Node(2, 2), Node(0, 2))
		self.assertEqual(p, L_2v2)

		# 2v1 transitions
		p = compute_probability(Node(2, 1), Node(2, 0))
		self.assertEqual(p, W_2v1)

		p = compute_probability(Node(2, 1), Node(1, 1))
		self.assertEqual(p, L_2v1)

		# 1v2 transitions
		p = compute_probability(Node(1, 10), Node(1, 9))
		self.assertEqual(p, W_1v2)

		p = compute_probability(Node(1, 10), Node(0, 10))
		self.assertEqual(p, L_1v2)

		# 1v1 transitions
		p = compute_probability(Node(1, 1), Node(1, 0))
		self.assertEqual(p, W_1v1)

		p = compute_probability(Node(1, 1), Node(0, 1))
		self.assertEqual(p, L_1v1)

	def test_compound_paths(self):
		W_3v2, T_3v2, L_3v2 = probability_space(Node(3, 2))
		W_3v1, T_3v1, L_3v1 = probability_space(Node(3, 1))
		W_2v2, T_2v2, L_2v2 = probability_space(Node(2, 2))
		W_2v1, T_2v1, L_2v1 = probability_space(Node(2, 1))
		W_1v2, T_1v2, L_1v2 = probability_space(Node(1, 2))
		W_1v1, T_1v1, L_1v1 = probability_space(Node(1, 1))

		# No ties with one dice
		self.assertEqual(T_3v1, Fraction(0))
		self.assertEqual(T_2v1, Fraction(0))
		self.assertEqual(T_1v1, Fraction(0))
		self.assertEqual(T_1v2, Fraction(0))

		p = compute_probability(Node(4, 2), Node(2, 1))
		expected = (
			T_3v2 * L_3v1
		)
		self.assertEqual(p, expected)

		p = compute_probability(Node(3, 3), Node(1, 1))
		expected = (
			T_2v2 * T_3v2 +
			L_2v1 * L_3v1 * W_3v2 +
			W_1v2 * W_1v2 * L_3v2
		)

	def test_total_one(self):
		one = paths_union(Node(75, 10), [
			# Don't include (0, 0), not valid anyways
			*(Node(0, i) for i in range(1, 11)),
			*(Node(i, 0) for i in range(1, 76)),
		])
		self.assertEqual(one, Fraction(1))

def print_probability_table():
	rows = []
	for a in range(3, 0, -1):
		for d in range(2, 0, -1):
			space = probability_space(Node(a, d))

			w_pct = f" (~{round(float(space.P_W) * 100)}%)" if space.W > 0 else ""
			t_pct = f" (~{round(float(space.P_T) * 100)}%)" if space.T > 0 else ""
			l_pct = f" (~{round(float(space.P_L) * 100)}%)" if space.L > 0 else ""

			rows.append([
				str(space.name),
				f"{space.W}{w_pct}",
				f"{space.T}{t_pct}",
				f"{space.L}{l_pct}",
				str(space.N)
			])

	# Calculate max width for each column
	widths = [max(len(row[i]) for row in rows) for i in range(5)]
	headers = ["Node", "W", "T", "L", "N"]
	widths = [max(widths[i], len(headers[i])) for i in range(5)]

	# Print header
	print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
	print("|" + "|".join("-" * (w + 2) for w in widths) + "|")

	# Print rows
	for row in rows:
		print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(5)) + " |")

def from_75_10_losing_n() -> dict[int, Fraction]:
	"""
	Generate an exact probability table for starting with (75, 10) and losing n troops
	This method most closely mirrors monte carlos simulations
	"""
	output: dict[int, Fraction] = {}
	# Compute victory scenarios
	for attackers in range(1, 76):
		output[75 - attackers] += compute_probability(Node(75, 10), Node(attackers, 0))
	# Compute defeat scenarios
	for defenders in range(1, 11):
		output[75] += compute_probability(Node(75, 10), Node(0, defenders))
	return output

def main():
	unittest.main()

	print_probability_table()

	scales = [
		(1e15, "100 trillion"), (1e14, "10 trillion"), (1e13, "a trillion"),
		(1e12, "100 billion"),  (1e10, "10 billion"),  (1e9,  "a billion"),
		(1e8,  "100 million"),  (1e7,  "10 million"),  (1e6,  "a million"),
		(1e5,  "100 thousand"), (1e4,  "10 thousand"), (1e3,  "a thousand"),
		(1e2, "a hundred")
	]

	probability_map = {i: paths_union(Node((i, 10)), [Node((2, n)) for n in range(1, 11)]) for i in range(50, 76)}
	for i in range(50, 76):
		p = float(probability_map[i])
		order = next((label for value, label in scales if p < 1/value), "<100")
		print(f"p({i} lost): {p:.2e}, worse than 1 in {order}")

if __name__ == '__main__':
	main()
