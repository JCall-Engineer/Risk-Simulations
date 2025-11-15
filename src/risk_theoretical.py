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

	def is_valid(self):
		"""(0, 0) is invalid because no risk battle results in both attackers and defenders getting wiped out"""
		return not any(i < 0 for i in [self.attackers, self.defenders]) and self.attackers + self.defenders > 0

	def has_edges(self):
		"""Acts as a check for a valid probability space"""
		return self.attackers > 0 and self.defenders > 0

	@property
	def space(self) -> ProbabilitySpace:
		return probability_space(self)

	def __hash__(self):
		"""Allows this class to be used as a dict key"""
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

# probability_space is going to be called *a lot* so cache the results
computed_spaces: dict[Node, ProbabilitySpace] = {}

# Counts all possible dice rolls and divides them into W L and T}
def probability_space(attackers: int | tuple[int, int] | Node = 3, defenders: int = 2) -> ProbabilitySpace:
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

# Compute the probability of transitioning from `start` to `end` using combinatorial calculations
def constant_space_probability(start: Node, end: Node) -> Fraction:
	if (any(i < 0 for i in [start.attackers, start.defenders, end.attackers, end.defenders])):
		raise ValueError(f"Negative troop counts are invalid: start={start}, end={end}")

	if (any(i == 0 for i in [start.attackers, start.defenders])):
		raise ValueError(f"There are no dice to be rolled at {start}")
	if (all(i == 0 for i in [end.attackers, end.defenders])):
		raise ValueError(f"{end} is not a state that can be reached")

	space = start.space
	if (space != end.space):
		raise ValueError(f"The number of dice used changes between {start} and {end}, this function is unable to compute probability over a non-uniform probability space")

	delta = start - end
	if (delta.attackers > 0 or delta.defenders > 0):
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
	# Handle terminal states where combat ends so we don't try and compute probability_space with 0 dice
	if end.attackers == 0 or end.defenders == 0:
		if end.attackers == 0 and end.defenders == 0:
			return Fraction(0)  # Both can't reach 0

		# We need to sum probability of all paths that lead to this terminal state
		# A terminal boundary is the last node before reaching 0
		total_probability = Fraction(0)

		if end.defenders == 0:
			for a in range(end.attackers, start.attackers + 1):
				for d in [1, 2]: # First consider nodes that hit this end from losing 1 defender
					if d == 2 and a < 2: # Then consider nodes that hit this end from losing 2 defenders
						continue
					terminal_node = Node(a, d)
					if terminal_node.attackers <= start.attackers and terminal_node.defenders <= start.defenders:
						prob_to_terminal = compute_probability(start, terminal_node)
						# From terminal_node, must win to reach this end
						total_probability += prob_to_terminal * terminal_node.space.P_W
		else:  # end.attackers == 0
			for d in range(end.defenders, start.defenders + 1):
				for a in [1, 2]: # First consider nodes that hit this end from losing 1 attackers
					if a == 2 and d < 2: # Then consider nodes that hit this end from losing 2 attackers
						continue
					terminal_node = Node(a, d)
					if terminal_node.defenders <= start.defenders and terminal_node.attackers <= start.attackers:
						prob_to_terminal = compute_probability(start, terminal_node)
						# From terminal_node, must lose to reach end
						total_probability += prob_to_terminal * terminal_node.space.P_L

		return total_probability

	# Handle the simple case early: both nodes in same probability space
	if start.space == end.space:
		return constant_space_probability(start, end)

	# Complex case: path crosses multiple probability spaces (e.g., 3v2 -> 2v2 -> 1v1)
	# Track cumulative probability of reaching each intermediary node across boundaries
	reach_probability: dict[Node, Fraction] = defaultdict(lambda: Fraction(0))

	# Identify all boundary nodes where at least one edge crosses into a different space
	space_by_dice = [(3, 2), (2, 2), (3, 1), (2, 1), (1, 2), (1, 1)] # pre-sorted by dice descending, attackers.descending, defenders descending
	boundaries_by_dice = {space: [] for space in space_by_dice}

	# Generate all boundary nodes that could be between start and end
	for a in range(end.attackers, start.attackers + 1):
		for d in range(end.defenders, start.defenders + 1):
			node = Node(a, d)
			if node.attackers == 0 or node.defenders == 0:
				continue
			space = (min(3, node.attackers), min(2, node.defenders))
			# Only track nodes at the boundary of their space
			boundary = ProbabilitySpaceBoundary(node)
			if boundary.is_boundary():
				boundaries_by_dice[space].append(boundary)

	# Process boundaries in descending dice order (3v2 first, then 2v2, etc.)
	# This ensures we compute reach probabilities before they're needed
	for space in space_by_dice:
		for boundary in boundaries_by_dice[space]:
			# Calculate probability of reaching this boundary node
			if boundary.node.space == start.space:
				reach_probability[boundary.node] = constant_space_probability(start, boundary.node)
			else:
				# Sum contributions from all previously-processed nodes in the same space
				# (constant_space_probability handles paths within a uniform space)
				for prev_node, prev_prob in reach_probability.items():
					if prev_prob > 0 and prev_node.space == boundary.node.space:
						reach_probability[boundary.node] += prev_prob * constant_space_probability(prev_node, boundary.node)

			# Propagate probability across space boundaries
			# Only edges that cross into a different space need explicit handling
			# Edges that do not cross into a different space are counted by the multinomial coefficient of the other boundary nodes they lead to
			node_prob = reach_probability[boundary.node]
			for outcome, edge_probability in boundary.traverse_edges():
				if outcome.space != boundary.node.space:
					reach_probability[outcome] += node_prob * edge_probability

	total_probability = Fraction(0)
	for node, step_probability in reach_probability.items():
		if step_probability > 0 and node.space == end.space:
			total_probability += step_probability * constant_space_probability(node, end)
		elif step_probability > 0:
			# Check if any edges from this node reach end's space
			boundary = ProbabilitySpaceBoundary(node)
			for outcome, edge_probability in boundary.traverse_edges():
				if outcome.space == end.space:
					total_probability += step_probability * edge_probability * constant_space_probability(outcome, end)
	return total_probability

# Sum multiple paths in the hypothetical DAG together for a composite probability
def paths_union(start: Node, ends: list[Node]):
	total_prob = Fraction(0)
	for end in ends:
		total_prob += compute_probability(start, end)
	return total_prob

import unittest
class TestProbabilities(unittest.TestCase):
	def test_edge_crossing(self):
		p = compute_probability(Node(75, 1), Node(75, 0))
		expected = Fraction(15, 36)
		self.assertEqual(p, expected)

		p = compute_probability(Node(75, 2), Node(75, 0))
		expected = Fraction(885, 1296)
		self.assertEqual(p, expected)

		p = compute_probability(Node(75, 2), Node(74, 1))
		expected = Fraction(2611, 7776)
		self.assertEqual(p, expected)

		p = compute_probability(Node(4, 2), Node(0, 2))
		expected = Fraction(2890, 7776) * Fraction (581, 1296)
		self.assertEqual(p, expected)

		p = compute_probability(Node(2, 2), Node(0, 2))
		expected = Fraction(2890, 7776) * Fraction (161, 216)
		self.assertEqual(p, expected)

		p = compute_probability(Node(3, 3), Node(0, 2))
		expected = (
			Fraction(2275, 7776) * Fraction (55, 216) * Fraction(161, 216) +
			Fraction(2611, 7776) * Fraction (581, 1296)
		)
		self.assertEqual(p, expected)

	def test_equals_one(self):
		one = paths_union(Node(75, 10), [
			*(Node(0, i) for i in range(1, 11)), # Start at 1 to avoid double counting (0, 0)
			*(Node(i, 0) for i in range(0, 76)),
		])
		self.assertEqual(one, Fraction(1))

def main():
	# What range of probabilities we are interested in considering
	lost_range = range(50, 76)
	exact_probability_of_losses  = {i: Fraction(0) for i in lost_range} # Stores the probability of losing n troops including the tail end from <5 dice rolls
	approx_probability_of_losses = {i: Fraction(0) for i in lost_range} # Stores the probability of losing n troops ignoring the tail end from <5 dice rolls

	scales = [
		(1e15, "100 trillion"), (1e14, "10 trillion"), (1e13, "a trillion"),
		(1e12, "100 billion"),  (1e10, "10 billion"),  (1e9,  "a billion"),
		(1e8,  "100 million"),  (1e7,  "10 million"),  (1e6,  "a million"),
		(1e5,  "100 thousand"), (1e4,  "10 thousand"), (1e3,  "a thousand"),
		(1e2, "a hundred")
	]

	for i in range(50, 76):
		p = float(exact_probability_of_losses[i])
		order = next((label for value, label in scales if p < 1/value), "<100")
		print(f"p({i} lost): {p:.2e}, worse than 1 in {order}")

if __name__ == '__main__':
	unittest.main()
