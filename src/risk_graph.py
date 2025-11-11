from graphviz import Digraph
import os

dot = Digraph(comment='Risk Battle DAG')
dot.attr(rankdir='TB')
dot.attr('node', fontsize='10')
dot.attr('edge', fontsize='9')
dot.attr(dpi='300')

# Helper to create vertical fraction label with better spacing
def frac_label(label, num, denom):
	return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1"><TR><TD>{label}:&nbsp;</TD><TD><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1"><TR><TD ALIGN="CENTER" BALIGN="CENTER">{num}</TD></TR><TR><TD BORDER="1" SIDES="B"></TD></TR><TR><TD ALIGN="CENTER" BALIGN="CENTER">{denom}</TD></TR></TABLE></TD></TR></TABLE>>'

# Define probabilities
probs = {
	(3, 2): {'W': (2890, 7776), 'T': (2611, 7776), 'L': (2275, 7776)},
	(3, 1): {'W': (885, 1296), 'T': (0, 1296), 'L': (441, 1296)},
	(2, 2): {'W': (295, 1296), 'T': (420, 1296), 'L': (581, 1296)},
	(2, 1): {'W': (125, 216), 'T': (0, 216), 'L': (91, 216)},
	(1, 2): {'W': (55, 216), 'T': (0, 216), 'L': (161, 216)},
	(1, 1): {'W': (15, 36), 'T': (0, 36), 'L': (21, 36)},
}

# Control spacing
dot.attr('graph', ranksep='0.5', nodesep='0.3')

# Group nodes by "level"
with dot.subgraph() as s:
	s.attr(rank='same')
	s.node('4_2', '(4, 2)', color='red', penwidth='2')
	s.node('3_3', '(3, 3)', color='red', penwidth='2')
	s.node('5_2', '(5, 2)', color='red', penwidth='2')

with dot.subgraph() as s:
	s.attr(rank='same')
	s.node('4_1', '(4, 1)', color='orange')
	s.node('3_2', '(3, 2)', color='orange')

with dot.subgraph() as s:
	s.attr(rank='same')
	s.node('4_0', '(4, 0)', color='green', style='filled', fillcolor='lightgreen')
	s.node('3_1', '(3, 1)', color='blue')
	s.node('2_2', '(2, 2)', color='blue')
	s.node('1_3', '(1, 3)', color='blue')
	s.node('5_0', '(5, 0)', color='green', style='filled', fillcolor='lightgreen')

with dot.subgraph() as s:
	s.attr(rank='same')
	s.node('3_0', '(3, 0)', color='green', style='filled', fillcolor='lightgreen')
	s.node('2_1', '(2, 1)', color='blue')
	s.node('1_2', '(1, 2)', color='blue')

with dot.subgraph() as s:
	s.attr(rank='same')
	s.node('2_0', '(2, 0)', color='green', style='filled', fillcolor='lightgreen')
	s.node('1_1', '(1, 1)', color='blue')
	s.node('0_2', '(0, 2)', color='red', style='filled', fillcolor='pink')
	s.node('1_0', '(1, 0)', color='green', style='filled', fillcolor='lightgreen')

with dot.subgraph() as s:
	s.attr(rank='same')
	s.node('0_1', '(0, 1)', color='red', style='filled', fillcolor='pink')
	s.node('0_3', '(0, 3)', color='red', style='filled', fillcolor='pink')

# Transitions from 5-dice states
p_42 = probs[(3, 2)]
dot.edge('4_2', '4_0', label=frac_label('W', p_42['W'][0], p_42['W'][1]))
dot.edge('4_2', '3_1', label=frac_label('T', p_42['T'][0], p_42['T'][1]))
dot.edge('4_2', '2_2', label=frac_label('L', p_42['L'][0], p_42['L'][1]))

p_33 = probs[(3, 2)]
dot.edge('3_3', '3_1', label=frac_label('W', p_33['W'][0], p_33['W'][1]))
dot.edge('3_3', '2_2', label=frac_label('T', p_33['T'][0], p_33['T'][1]))
dot.edge('3_3', '1_3', label=frac_label('L', p_33['L'][0], p_33['L'][1]))

p_52 = probs[(3, 2)]
dot.edge('5_2', '5_0', label=frac_label('W', p_52['W'][0], p_52['W'][1]))
dot.edge('5_2', '4_1', label=frac_label('T', p_52['T'][0], p_52['T'][1]))
dot.edge('5_2', '3_2', label=frac_label('L', p_52['L'][0], p_52['L'][1]))

# Transitions from 4-dice states (3v1)
p_31 = probs[(3, 1)]
dot.edge('3_2', '3_0', label=frac_label('W', p_31['W'][0], p_31['W'][1]))
dot.edge('3_2', '1_2', label=frac_label('L', p_31['L'][0], p_31['L'][1]))

dot.edge('4_1', '4_0', label=frac_label('W', p_31['W'][0], p_31['W'][1]))
dot.edge('4_1', '2_1', label=frac_label('L', p_31['L'][0], p_31['L'][1]))

dot.edge('3_1', '3_0', label=frac_label('W', p_31['W'][0], p_31['W'][1]))
dot.edge('3_1', '1_1', label=frac_label('L', p_31['L'][0], p_31['L'][1]))

# Transitions from 4-dice states (2v2)
p_22 = probs[(2, 2)]
dot.edge('2_2', '2_0', label=frac_label('W', p_22['W'][0], p_22['W'][1]))
dot.edge('2_2', '1_1', label=frac_label('T', p_22['T'][0], p_22['T'][1]))
dot.edge('2_2', '0_2', label=frac_label('L', p_22['L'][0], p_22['L'][1]))

# Transitions from 3-dice states (2v1)
p_21 = probs[(2, 1)]
dot.edge('2_1', '2_0', label=frac_label('W', p_21['W'][0], p_21['W'][1]))
dot.edge('2_1', '0_1', label=frac_label('L', p_21['L'][0], p_21['L'][1]))

# Transitions from 3-dice states (1v2)
p_12 = probs[(1, 2)]
dot.edge('1_2', '1_0', label=frac_label('W', p_12['W'][0], p_12['W'][1]))
dot.edge('1_2', '0_2', label=frac_label('L', p_12['L'][0], p_12['L'][1]))

dot.edge('1_3', '1_1', label=frac_label('W', p_12['W'][0], p_12['W'][1]))
dot.edge('1_3', '0_3', label=frac_label('L', p_12['L'][0], p_12['L'][1]))

# Transitions from 2-dice states (1v1)
p_11 = probs[(1, 1)]
dot.edge('1_1', '1_0', label=frac_label('W', p_11['W'][0], p_11['W'][1]))
dot.edge('1_1', '0_1', label=frac_label('L', p_11['L'][0], p_11['L'][1]))

os.makedirs('out', exist_ok=True)
dot.render('out/risk_dag', format='png', view=True, cleanup=True)
print("Graph saved as out/risk_dag.png")