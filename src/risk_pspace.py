from matplotlib import pyplot, patches
import numpy as np

figure, axes = pyplot.subplots(figsize=(14, 10))

# Define circles
attackers_to_x = {
	75: 1,
	4: 4,
	3: 5,
	2: 6.5,
	1: 8,
	0: 9.5,
}
defenders_to_y = {
	10: 10,
	3: 8,
	2: 7,
	1: 5.5,
	0: 4,
}
circles = [
	(75, 10), (75, 3), (75, 2), (75, 1), (75, 0),
	(4, 10),           (4, 2),
	(3, 10),  (3, 3),  (3, 2),  (3, 1),  (3, 0),
	(2, 10),  (2, 3),  (2, 2),  (2, 1),  (2, 0),
	(1, 10),           (1, 2),  (1, 1),  (1, 0),
	(0, 10),           (0, 2),  (0, 1),
]

# Circle positions: {label: (x, y)}
circle_xy = {f'{c[0]},{c[1]}': (attackers_to_x[c[0]], defenders_to_y[c[1]]) for c in circles}
radius = 0.3

# Draw circles
for label, (x, y) in circle_xy.items():
	circle = patches.Circle((x, y), radius, color='lightblue', ec='black', linewidth=2)
	axes.add_patch(circle)
	axes.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')

def calculate_box(pixel_coords, padding=0.2):
	"""
	Calculate box boundaries from list of pixel coordinates.

	Args:
		pixel_coords: List of (x, y) pixel coordinate tuples
		padding: Extra space around the circles

	Returns:
		(x, y, width, height) tuple for the box
	"""
	x_positions = [coord[0] for coord in pixel_coords]
	y_positions = [coord[1] for coord in pixel_coords]

	min_x = min(x_positions) - radius - padding
	max_x = max(x_positions) + radius + padding
	min_y = min(y_positions) - radius - padding
	max_y = max(y_positions) + radius + padding

	width = max_x - min_x
	height = max_y - min_y

	return (min_x, min_y, width, height)

# Define boxes by their circle string keys
box_definitions = {
	'3v2': ['75,10', '75,3', '75,2', '4,10', '4,2', '3,10', '3,3', '3,2'],
	'2v2': ['2,10', '2,3', '2,2'],
	'1v2': ['1,10', '1,2'],
	'3v1': ['75,1', '3,1'],
	'': ['2,1'],
	' ': ['1,1'],
	'defeat': ['0,10', '0,2', '0,1'],
	'victory': ['75,0', '3,0', '2,0', '1,0']
}

# Calculate boxes
boxes = {label: calculate_box([circle_xy[key] for key in keys]) for label, keys in box_definitions.items()}

# Draw boxes
for label, (x, y, w, h) in boxes.items():
	rectangle = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
	axes.add_patch(rectangle)
	if label and label not in ['', ' ']:
		axes.text(x + w/2, y + h + 0.05, label, ha='center', va='bottom', fontsize=12, weight='bold')

# Helper function to get edge points
def get_edge_points(start, end):
	x1, y1 = circle_xy[start]
	x2, y2 = circle_xy[end]
	dx = x2 - x1
	dy = y2 - y1
	dist = np.sqrt(dx**2 + dy**2)
	if dist == 0:
		return x1, y1, x2, y2
	dx_norm = dx / dist
	dy_norm = dy / dist
	x1_edge = x1 + radius * dx_norm
	y1_edge = y1 + radius * dy_norm
	x2_edge = x2 - radius * dx_norm
	y2_edge = y2 - radius * dy_norm
	return x1_edge, y1_edge, x2_edge, y2_edge

# Straight connections with dots
connections = [
	# 3v2 box connections
	('75,10', '4,10'),
	('4,10', '3,10'),

	('3,10', '3,3'),
	('3,3', '3,2'),

	('75,10', '75,3'),
	('75,3', '75,2'),

	('75,2', '4,2'),
	('4,2', '3,2'),

	('75,10', '3,2'),

	# 3v1 box
	('75,1', '3,1'),

	# 2v2 box
	('2,10', '2,3'),
	('2,3', '2,2'),

	# 1v2 box
	('1,10', '1,2'),

	# Between Boxes
	('75,2', '75,1'),
	('75,1', '75,0'),

	('3,1', '3,0'),
	('3,1', '2,1'),

	('2,1', '1,1'),
	('2,1', '2,0'),

	('1,1', '0,1'),
	('1,1', '1,0'),

	('1,10', '0,10'),

	('1,2', '0,2'),
	('1,2', '1,1'),

	('3,2', '2,1'),

	('3,3', '2,2'),
	('4,2', '3,1'),
	('2,3', '1,2'),
	('2,2', '1,1'),
]

for start, end in connections:
	x1, y1, x2, y2 = get_edge_points(start, end)
	axes.plot([x1, x2], [y1, y2], 'k:', linewidth=1.5, marker='o', markersize=3)

# Curved connections
def draw_curve(start, end, axes, direction='auto'):
	x1, y1 = circle_xy[start]
	x2, y2 = circle_xy[end]

	# Create a curved path
	t = np.linspace(0, 1, 100)
	mid_x = (x1 + x2) / 2
	mid_y = (y1 + y2) / 2

	# Add curvature perpendicular to the line
	dx = x2 - x1
	dy = y2 - y1

	# Adjust direction based on parameter
	match direction:
		case 'left':
			perp_x = dy * 0.3
			perp_y = -dx * 0.3
		case 'right':
			perp_x = -dy * 0.3
			perp_y = dx * 0.3
		case 'up':
			perp_x = dy * 0.3
			perp_y = dx * 0.3
		case 'down':
			perp_x = -dy * 0.3
			perp_y = -dx * 0.3
		case _:  # 'auto' or any other value
			perp_x = -dy * 0.3
			perp_y = dx * 0.3

	x_curve = (1-t)**2 * x1 + 2*(1-t)*t * (mid_x + perp_x) + t**2 * x2
	y_curve = (1-t)**2 * y1 + 2*(1-t)*t * (mid_y + perp_y) + t**2 * y2

	# Trim to edges
	dist_start = np.sqrt((x_curve - x1)**2 + (y_curve - y1)**2)
	dist_end = np.sqrt((x_curve - x2)**2 + (y_curve - y2)**2)
	mask = (dist_start >= radius) & (dist_end >= radius)
	axes.plot(x_curve[mask], y_curve[mask], 'k:', linewidth=1.5)

curved_connections = [
	('4,10', '2,10', 'up'),
	('3,10', '1,10', 'up'),
	('4,2', '2,2', 'down'),
	('3,2', '1,2', 'down'),
	('3,3', '3,1', 'right'),
	('3,2', '3,0', 'right'),
	('2,2', '2,0', 'left'),
	('2,2', '0,2', 'down'),
	('2,10', '0,10', 'up'),
]

for conn in curved_connections:
	if len(conn) == 3:
		draw_curve(conn[0], conn[1], axes, conn[2])
	else:
		draw_curve(conn[0], conn[1], axes)

axes.set_xlim(0, 10.25)
axes.set_ylim(3.25, 11)
axes.set_aspect('equal')
axes.axis('off')
pyplot.suptitle('Risk Dice Rolling Scenarios', fontsize=16, weight='bold')
pyplot.tight_layout()

figure.patch.set_alpha(0)
pyplot.savefig('out/risk_pspace.png', dpi=300, bbox_inches='tight', facecolor='none', pad_inches=0.5)
pyplot.show()
