import numpy
import cupy
from datetime import datetime
from utils.params import SimulationParams, monitor_progress

def run_simulation_cuda(params: SimulationParams) -> numpy.ndarray:
	BINS = params.battle.attackers + 1
	kernel_code = f"""
#define N_SIMULATIONS {params.simulations}
#define ATTACKERS {params.battle.attackers}
#define DEFENDERS {params.battle.defenders}
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
	assert THREADS_PER_BLOCK >= params.battle.attackers # Make sure it is greater than attackers for local bin optimization
	BLOCKS_NEEDED = params.simulations // THREADS_PER_BLOCK + (params.simulations % THREADS_PER_BLOCK > 0)
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
