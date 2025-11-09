import sys
import time
import numpy
import cupy
import multiprocessing
from multiprocessing import Manager
from multiprocessing.synchronize import Lock
from multiprocessing.managers import ValueProxy
from matplotlib import pyplot
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Simulation:
	Attackers: int
	Defenders: int
	Simulations: int

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

# multithreading doesn't handle captured variables or nested functions well, sad
def simulate_battle(job: Simulation, rng: numpy.random.Generator):
		attackers_left = job.Attackers
		defenders_left = job.Defenders
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
		return job.Attackers - attackers_left

def worker(job:Simulation, batch_size: int, update_interval: int, shared_progress: ValueProxy, lock: Lock) -> numpy.ndarray:
	rng = numpy.random.default_rng()
	results = numpy.empty(batch_size, dtype=numpy.int32)

	last_reported = -1 # Init at 0 would imply we reported index 0
	for i in range(batch_size):
		results[i] = simulate_battle(job, rng)
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
def run_simulation_cpu(job: Simulation):
	bins = job.Attackers + 1
	num_workers = multiprocessing.cpu_count()
	batch_size = job.Simulations // num_workers
	remainder = job.Simulations % num_workers
	update_interval = max(1, batch_size // 100)

	# Handle remainder simulations in main process
	remainder_results = numpy.empty(remainder, dtype=numpy.int32)
	rng = numpy.random.default_rng()
	for i in range(remainder):
		remainder_results[i] = simulate_battle(job, rng)

	manager = Manager()
	shared_progress = manager.Value('i', remainder)
	lock = manager.Lock()

	with multiprocessing.Pool(num_workers) as pool:
		results = pool.starmap_async(worker, [(job, batch_size, update_interval, shared_progress, lock)] * num_workers)
		monitor_progress(get_progress=lambda: shared_progress.value, total=job.Simulations)
		print("\nCPU Simulation complete!")
		all_results = numpy.concatenate([remainder_results, *results.get()])

	return numpy.bincount(all_results, minlength=bins)

def run_simulation_cuda(job: Simulation):
	BINS = job.Attackers + 1
	kernel_code = f"""
#define N_SIMULATIONS {job.Simulations}
#define ATTACKERS {job.Attackers}
#define DEFENDERS {job.Defenders}
#define BINS {BINS}
""" + """
#include <curand_kernel.h>

extern "C" __global__
void risk_battle_kernel(uint64_t *results, uint64_t *progress, int seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= N_SIMULATIONS) return;

	// Initialize bins to reduce memory accesses
	__shared__ uint64_t local_results[BINS];
	if (threadIdx.x < BINS) local_results[threadIdx.x] = 0;
	__syncthreads();

	// RNG state per thread
	curandStateMRG32k3a state;
	curand_init(seed + idx * 12345, idx, 0, &state);

	int attackers_left = ATTACKERS;
	int defenders_left = DEFENDERS;

	while (attackers_left > 0 && defenders_left > 0) {
		int attack_rolls[3];
		int defend_rolls[2];

		for (int i = 0; i < min(3, attackers_left); i++) {
			attack_rolls[i] = 1 + (int)(curand(&state) % 6);
		}
		for (int i = 0; i < min(2, defenders_left); i++) {
			defend_rolls[i] = 1 + (int)(curand(&state) % 6);
		}

		// simple bubble sort for up to 3 elements (descending)
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2 - i; j++) {
				if (attack_rolls[j] < attack_rolls[j + 1]) {
					int tmp = attack_rolls[j];
					attack_rolls[j] = attack_rolls[j + 1];
					attack_rolls[j + 1] = tmp;
				}
			}
		}
		// defense 2-element sort
		if (defend_rolls[0] < defend_rolls[1]) {
			int tmp = defend_rolls[0];
			defend_rolls[0] = defend_rolls[1];
			defend_rolls[1] = tmp;
		}

		// resolve
		for (int i = 0; i < min(3, 2); i++) {
			if (attack_rolls[i] > defend_rolls[i]) defenders_left -= 1;
			else attackers_left -= 1;
		}
	}

	int lost = ATTACKERS - max(0, attackers_left);

	// local histogram in shared memory
	atomicAdd(&local_results[lost], 1ULL);
	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = 0; i < BINS; ++i) {
			atomicAdd(&results[i], local_results[i]);
		}
		// increment the single global progress counter by 1 (one per block)
		atomicAdd(&progress[0], 1ULL);
	}
}
"""
	# compile kernels
	risk_battle_kernel = cupy.RawKernel(kernel_code, "risk_battle_kernel")

	# launch config
	block_dim = (256, 1, 1)
	THREADS_PER_BLOCK = block_dim[0]
	assert THREADS_PER_BLOCK >= job.Attackers # Make sure it is greater than attackers for local bin optimization
	BLOCKS_NEEDED = job.Simulations // THREADS_PER_BLOCK + (job.Simulations % THREADS_PER_BLOCK > 0)
	grid_dim = (BLOCKS_NEEDED, 1, 1)

	# datetime based seed masked to 32 bits to ensure correct packing
	seed = int(int(datetime.now().timestamp() * 1e6) & 0xFFFFFFFF)

	# device allocations
	d_results = cupy.zeros(BINS, dtype=cupy.uint64)
	# single 8-byte progress counter (cheap to copy)
	d_progress = cupy.zeros(1, dtype=cupy.uint64)

	# launch
	stream = cupy.cuda.Stream(non_blocking=True)
	with stream:
		risk_battle_kernel(grid_dim, block_dim, (d_results, d_progress, seed))

	# monitor using cheap 8-byte copy
	monitor_progress(get_progress=lambda: int(d_progress.get()[0]), total=BLOCKS_NEEDED)

	stream.synchronize()
	print("\nCUDA Simulation complete!")
	return d_results.get()

def print_data(task: Simulation, results: numpy.ndarray, label: str):
	print(f"{label} distribution:")
	for i, count in enumerate(results):
		print(f"{i:>2} attackers lost {count:>15,}", end='  |  ' if (i + 1) % 5 else '\r\n')
	check = numpy.sum(results)
	print(f" Sanity Check - total events: {check:>15,} ({'correct' if check == task.Simulations else 'incorrect'})")

# Run the simulations
if __name__ == "__main__":
	task = Simulation(
		Attackers=75,
		Defenders=10,
		Simulations=10_000_000_000,
	)
	results_cuda = run_simulation_cuda(task)
	print_data(task, results_cuda, 'CUDA')

	task.Simulations = 100_000_000 # My CPU cannot handle 1 billion simulations in a reasonable timeframe
	results_cpu = run_simulation_cpu(task)
	print_data(task, results_cpu, 'CPU')

	# Plot histograms
	pyplot.bar(range(len(results_cuda)), results_cuda, color='#76B900', alpha=0.5, label='CUDA')
	pyplot.bar(range(len(results_cpu)),  results_cpu,  color='#ED1C24', alpha=0.5, label='CPU')
	pyplot.xlabel("Attackers Lost")
	pyplot.ylabel("Frequency")
	pyplot.title("Risk Battle Simulation Results")
	pyplot.legend()
	pyplot.show()
