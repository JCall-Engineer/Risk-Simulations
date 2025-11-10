import numpy
import cupy
from datetime import datetime
from utils.params import SimulationParams, BattleParams, monitor_progress, simulations_value, time_str

def make_kernel_and_run(block_size: int, params: SimulationParams) -> tuple[float, numpy.ndarray]:
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

	kernel_code = f"""
#define N_SIMULATIONS {params.simulations}
#define ATTACKERS {params.battle.attackers}
#define DEFENDERS {params.battle.defenders}
#define WARPS {WARPS}
#define BINS {BINS}
""" + """
#include <curand_kernel.h>

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

	int lost = ATTACKERS - attackers_left;

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
	# compile kernels
	risk_battle_kernel = cupy.RawKernel(kernel_code, "risk_battle_kernel")

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
	return make_kernel_and_run(256, params)

def test_kernel_sizes():
	params = SimulationParams(
		battle=BattleParams(
			attackers=75,
			defenders=10,
		),
		simulations=simulations_value('100m'),
	)
	for BLOCK_SIZE in [256, 512, 1024]:
		time, results = make_kernel_and_run(BLOCK_SIZE, params)
		print(f"Block Size: {BLOCK_SIZE}, time: {time_str(time)}")
