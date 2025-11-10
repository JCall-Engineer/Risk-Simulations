import os
import re
import sys
import time
import numpy
from enum import Enum
from dataclasses import dataclass
from collections.abc import Callable

@dataclass
class BattleParams:
	"""Parameters for a single battle scenario"""
	attackers: int
	defenders: int

@dataclass
class SimulationParams:
	"""Parameters for a simulation request"""
	battle: BattleParams
	simulations: int

@dataclass
class ProcessorConfig:
	"""Parameters for a simulator profile"""
	label: str
	runner: Callable[[SimulationParams], numpy.ndarray]

class Processor(Enum):
	CPU = 'CPU'
	GPU = 'CUDA'

@dataclass
class SimulationRequest:
	"""Parameters for a simulation and which device should run it"""
	params: SimulationParams
	processor: Processor

def time_str(seconds):
	mins = int(seconds // 60)
	secs = seconds % 60 # Apparently this preserves the decimal in python
	return f"{mins}m {secs:.1f}s" if mins else f"{secs:.1f}s"

def monitor_progress(get_progress, total, poll=1.0):
	start_time = time.time()
	while True:
		elapsed = time.time() - start_time

		progress = get_progress()
		percent_complete = min((progress / total) * 100, 100)
		estimate = (elapsed * (100 - percent_complete) / percent_complete) if percent_complete > 0 else 0

		sys.stdout.write(f"\rProgress: {percent_complete:.2f}% - Elapsed: {time_str(elapsed)} - Remaining: {time_str(estimate)}     ")
		sys.stdout.flush()

		if progress >= total: return
		time.sleep(poll)

Magnitude = {
	'b': 1_000_000_000,
	'm': 1_000_000,
	'k': 1_000,
	 '': 1,
}
def simulations_label(value: int) -> str:
	for suffix, multiplier in Magnitude.items():
		if value >= multiplier and value % multiplier == 0:
			return f"{value // multiplier}{suffix}"
	return str(value)
def simulations_value(label: str) -> int:
	label = label.lower().strip()
	suffix = label[-1] if label and label[-1] in Magnitude else ''
	number = label[:-1] if suffix else label
	return int(number) * Magnitude[suffix]

def make_slug(job: SimulationRequest) -> str:
	return f"{job.processor.value}-N{simulations_label(job.params.simulations)}-A{job.params.battle.attackers}-D{job.params.battle.defenders}"
def parse_slug(file_name: str) -> SimulationRequest:
	pattern = rf'^(\w+)-N(\d+[{''.join(Magnitude.keys())}]?)-A(\d+)-D(\d+)'
	match = re.match(pattern, file_name)
	if not match:
		raise ValueError(f"Invalid slug format: {file_name}")
	processor_label, sims, attackers, defenders = match.groups()

	try:
		processor = next(p for p in Processor if p.value == processor_label)
	except StopIteration:
		raise ValueError(f"Unknown processor label: {processor_label}")

	return SimulationRequest(
		params=SimulationParams(
			battle=BattleParams(
				attackers=int(attackers),
				defenders=int(defenders)
			),
			simulations=simulations_value(sims)
		),
		processor=processor
	)

def save_results(file_name: str, results: numpy.ndarray) -> None:
	os.makedirs("out", exist_ok=True)
	with open(file_name, 'w') as f:
		for count in results:
			f.write(f"{count}\n")
	print(f"Saved results to {file_name}")
def load_results(slug: str) -> numpy.ndarray:
	file_name = f"out/{slug}.txt"
	return numpy.loadtxt(file_name, dtype=numpy.uint64)
def list_saves() -> list[SimulationRequest]:
	out: list[SimulationRequest] = []

	if not os.path.exists("out"):
		return out

	for file_name in os.listdir("out"):
		if not file_name.endswith('.txt'):
			continue
		try:
			out.append(parse_slug(file_name))
		except ValueError:
			continue  # Skip files that don't match the pattern

	return out

def print_results(job: SimulationRequest, results: numpy.ndarray) -> None:
	print(f"{job.processor.value} distribution:")
	for i, count in enumerate(results):
		print(f"{i:>2} attackers lost {count:>15,}", end='  |  ' if (i + 1) % 5 else '\r\n')
	check = numpy.sum(results)
	print(f"\nSanity Check - total events: {check:>15,} ({'correct' if check == job.params.simulations else 'incorrect'})")

def run_simulation(job: SimulationRequest) -> numpy.ndarray:
	from procs.sim_cpu import run_simulation_cpu
	from procs.sim_gpu import run_simulation_cuda
	slug = make_slug(job)
	file_name = f"out/{slug}.txt"
	if os.path.exists(file_name):
		confirm = input(f"Results exist for {slug}. Overwrite? (y/n): ").strip().lower()
		if confirm != 'y':
			print('Loading existing results...')
			return load_results(slug)

	print(f"Running {slug} simulation...")

	runner = {
		Processor.CPU: run_simulation_cpu,
		Processor.GPU: run_simulation_cuda,
	}[job.processor]

	results = runner(job.params)
	save_results(file_name, results)
	return results

def job(asking: Processor, simulations: int, attackers: int, defenders: int) -> SimulationRequest:
	return SimulationRequest(
		params=SimulationParams(
			battle=BattleParams(
				attackers=attackers,
				defenders=defenders
			),
			simulations=simulations
		),
		processor=asking
	)
