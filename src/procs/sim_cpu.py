import numpy
import multiprocessing
from multiprocessing import Manager
from multiprocessing.synchronize import Lock
from multiprocessing.managers import ValueProxy
from utils.params import BattleParams, SimulationParams, monitor_progress

# multithreading doesn't handle captured variables or nested functions well, sad
def simulate_battle(battle: BattleParams, rng: numpy.random.Generator):
		attackers_left = battle.attackers
		defenders_left = battle.defenders
		while attackers_left > 0 and defenders_left > 0:
			attack_rolls = rng.integers(1, 7, min(3, attackers_left))
			defend_rolls = rng.integers(1, 7, min(2, defenders_left))
			attack_rolls.sort()
			defend_rolls.sort()
			for a, d in zip(attack_rolls[::-1], defend_rolls[::-1]):
				if a > d:
					defenders_left -= 1
				else:
					attackers_left -= 1
		return battle.attackers - attackers_left

def worker(battle: BattleParams, batch_size: int, update_interval: int, shared_progress: ValueProxy, lock: Lock) -> numpy.ndarray:
	rng = numpy.random.default_rng()
	results = numpy.empty(batch_size, dtype=numpy.int32)

	last_reported = -1 # Init at 0 would imply we reported index 0
	for i in range(batch_size):
		results[i] = simulate_battle(battle, rng)
		unreported = i - last_reported
		if unreported >= update_interval:
			with lock:
				shared_progress.value += unreported
			last_reported = i

	# Update any remaining progress
	last_index = batch_size - 1
	if last_reported < last_index:
		with lock:
			shared_progress.value += last_index - last_reported

	return results

# CPU Simulation
def run_simulation_cpu(params: SimulationParams) -> numpy.ndarray:
	bins = params.battle.attackers + 1
	num_workers = multiprocessing.cpu_count()
	batch_size = params.simulations // num_workers
	remainder = params.simulations % num_workers
	update_interval = max(1, batch_size // 100)

	# Handle remainder simulations in main process
	remainder_results = numpy.empty(remainder, dtype=numpy.int32)
	rng = numpy.random.default_rng()
	for i in range(remainder):
		remainder_results[i] = simulate_battle(params.battle, rng)

	manager = Manager()
	shared_progress = manager.Value('i', remainder)
	lock = manager.Lock()

	with multiprocessing.Pool(num_workers) as pool:
		results = pool.starmap_async(worker, [(params.battle, batch_size, update_interval, shared_progress, lock)] * num_workers)
		monitor_progress(get_progress=lambda: shared_progress.value, total=params.simulations)
		print("\nCPU Simulation complete!")
		all_results = numpy.concatenate([remainder_results, *results.get()])

	return numpy.bincount(all_results, minlength=bins)
