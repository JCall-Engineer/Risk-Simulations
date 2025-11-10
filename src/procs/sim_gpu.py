import numpy
import cupy

from datetime import datetime

from enum import Enum
from dataclasses import dataclass

from utils.params import SimulationParams, BattleParams, monitor_progress, simulations_value, time_str

class CudaImplementation(Enum):
	Naive = """
__device__ inline int simulation(curandStateMRG32k3a& state) {
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

	return ATTACKERS - max(0, attackers_left);
}
"""
	FastPath = """
__device__ inline int simulation(curandStateMRG32k3a& state) {
	int attackers_left = ATTACKERS;
	int defenders_left = DEFENDERS;

	// FAST PATH: Handle the common case (3v2) with NO branches
	while (attackers_left > 2 && defenders_left > 1) {
		// Always roll 5 dice, always sort same way, always compare 2 battles
		unsigned int rand_val = curand(&state);
		int a1 = 1 + ((rand_val >> 0) & 0x7) % 6;
		int a2 = 1 + ((rand_val >> 3) & 0x7) % 6;
		int a3 = 1 + ((rand_val >> 6) & 0x7) % 6;
		int d1 = 1 + ((rand_val >> 9) & 0x7) % 6;
		int d2 = 1 + ((rand_val >> 12) & 0x7) % 6;

		// Branchless sorting network for 3 elements
		int tmp;
		tmp = max(a1, a2); a1 = min(a1, a2); a2 = tmp;
		tmp = max(a2, a3); a2 = min(a2, a3); a3 = tmp;
		tmp = max(a1, a2); a1 = min(a1, a2); a2 = tmp;

		// Sort defense (2 elements)
		tmp = max(d1, d2); d1 = min(d1, d2); d2 = tmp;

		// Branchless combat resolution
		defenders_left -= (a3 > d2);
		attackers_left -= (a3 <= d2);
		defenders_left -= (a2 > d1);
		attackers_left -= (a2 <= d1);
	}

	// SLOW PATH: Handle edge cases (1v1, 2v1, 3v1, 1v2, 2v2)
	while (attackers_left > 0 && defenders_left > 0) {
		// This has branches, but runs rarely (only last few rounds)
		int attack_count = min(3, attackers_left);
		int defend_count = min(2, defenders_left);

		unsigned int rand_val = curand(&state);
		int battles = min(attack_count, defend_count);

		for (int i = 0; i < battles; i++) {
			int a = 1 + ((rand_val >> (i*3)) & 0x7) % 6;
			int d = 1 + ((rand_val >> (9 + i*3)) & 0x7) % 6;
			defenders_left -= (a > d);
			attackers_left -= (a <= d);
		}
	}

	return ATTACKERS - attackers_left;
}
"""

@dataclass
class KernelParams:
	simulations: int
	attackers: int
	defenders: int
	warps: int
	bins: int

def kernel(params: KernelParams, implementation: CudaImplementation) -> str:
	return f"""
#define N_SIMULATIONS {params.simulations}
#define ATTACKERS {params.attackers}
#define DEFENDERS {params.defenders}
#define WARPS {params.warps}
#define BINS {params.bins}
#include <curand_kernel.h>
{implementation.value}""" + """
extern "C" __global__
void risk_battle_kernel(uint64_t *results, uint64_t *progress, int seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= N_SIMULATIONS) return;

	// Use warp-level shared bins for synchronicity guarantees
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	__shared__ uint64_t warp_results[WARPS][BINS];

	// Efficiently initialize warp_results
	int total_elements = WARPS * BINS;
	for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
		int warp_idx = i / BINS;
		int bin_idx = i % BINS;
		warp_results[warp_idx][bin_idx] = 0;
	}
	__syncthreads();

	// RNG state per thread
	curandStateMRG32k3a state;
	curand_init(seed + idx * 12345, idx, 0, &state);

	int lost = simulation(state);

	// Warp-level histogram
	atomicAdd(&warp_results[warp_id][lost], 1ULL);

	// Wait for all warps
	__syncthreads();

	// Single thread reduces to global memory (serialized atomics avoid contention)
	if (threadIdx.x == 0) {
		#pragma unroll // Eliminate loop overhead - reduction is fast, atomics are the bottleneck
		for (int i = 0; i < BINS; ++i) {
			uint64_t sum = 0;
			//#pragma unroll 8  // Balance unrolling benefit vs code size (limits I-cache pressure for large blocks)
			#pragma unroll // Let compiler choose unroll factor based on code size heuristics
			for (int w = 0; w < WARPS; ++w) {
				sum += warp_results[w][i];
			}
			if (sum > 0) {
				atomicAdd(&results[i], sum);
			}
		}
		atomicAdd(&progress[0], 1ULL);
	}
}
"""

def make_kernel_and_run(block_size: int, params: SimulationParams, implementation: CudaImplementation) -> tuple[float, numpy.ndarray]:
	if block_size % 32 != 0:
		raise ValueError("Block size must be an even multiple of warp size (32)")
	if block_size > 1024:
		raise ValueError("Block size cannot exceed 1024 (GPU hardware limit)")

	WARPS = block_size // 32 # A warp is 32 threads on my hardware
	BINS = params.battle.attackers + 1 # 0 lost up to and including attackers lost

	# launch config
	block_dim = (block_size, 1, 1) # We are solving a one dimensional problem (as opposed to images or 3d rendering) and don't gain anything from added complexity
	THREADS_PER_BLOCK = block_dim[0] * block_dim[1] * block_dim[2]
	BLOCKS_NEEDED = params.simulations // THREADS_PER_BLOCK + (params.simulations % THREADS_PER_BLOCK > 0)
	grid_dim = (BLOCKS_NEEDED, 1, 1)

	# compile kernels
	risk_battle_kernel = cupy.RawKernel(kernel(KernelParams(
		simulations=params.simulations,
		attackers=params.battle.attackers,
		defenders=params.battle.defenders,
		warps=WARPS,
		bins=BINS,
	), implementation), "risk_battle_kernel")

	# datetime based seed masked to 32 bits to ensure correct packing
	seed = int(int(datetime.now().timestamp() * 1e6) & 0xFFFFFFFF)

	# device allocations
	d_results = cupy.zeros(BINS, dtype=cupy.uint64)
	d_progress = cupy.zeros(1, dtype=cupy.uint64)

	# launch
	stream = cupy.cuda.Stream(non_blocking=True)
	with stream:
		risk_battle_kernel(grid_dim, block_dim, (d_results, d_progress, seed))

	# monitor using cheap 8-byte copy
	time = monitor_progress(get_progress=lambda: int(d_progress.get()[0]), total=BLOCKS_NEEDED)

	stream.synchronize()
	print("\nCUDA Simulation complete!")
	return time, d_results.get()

def run_simulation_cuda(params: SimulationParams) -> tuple[float, numpy.ndarray]:
	# Testing showed identical performance between 256 and 512, and not enough resources for 1024
	# The occasional usage drops I am seeing in task manager weren't a lack of parallelism but memory or atomics
	return make_kernel_and_run(256, params, CudaImplementation.FastPath)

def test_kernel_performance():
	"""
Output:
	"""
	params = SimulationParams(
		battle=BattleParams(
			attackers=75,
			defenders=10,
		),
		simulations=simulations_value('100m'),
	)

	results = []
	for implementation in CudaImplementation:
		for BLOCK_SIZE in [32, 64, 128, 256, 512, 1024]:
			print(f"\nTesting {implementation.name} with {BLOCK_SIZE} threads per block...")
			try:
				time, _ = make_kernel_and_run(BLOCK_SIZE, params, implementation)
				results.append((implementation.name, BLOCK_SIZE, time_str(time)))
			except cupy.cuda.driver.CUDADriverError as e:
				results.append((implementation.name, BLOCK_SIZE, "FAILED"))

	# Print summary table
	print("\n" + "=" * 40)
	print("RESULTS SUMMARY")
	print("=" * 40)
	print(f"{'Implementation':<15} | {'Block':>5} | {'Time':>8}")
	print("-" * 40)

	for impl_name, block_size, time in results:
		print(f"{impl_name:<15} | {block_size:>5} | {time:>8}")
