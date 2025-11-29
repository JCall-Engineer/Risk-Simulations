from matplotlib import pyplot
from utils.params import Processor, simulations_label, simulations_value, list_saves, print_results, run_simulation, job, make_slug, load_results
from procs.sim_gpu import test_kernel_performance
from procs.theoretical import from_start_losing_n

class RiskSimulator:
	def __init__(self):
		self.simulations = 1_000_000
		self.attackers = 75
		self.defenders = 10
		self.selected = []

	def edit_params(self):
		print("\nEdit Simulation Parameters (press Enter to keep current value)")

		input_simulations = input(f"Simulations [{simulations_label(self.simulations)}]: ").strip()
		if input_simulations:
			try:
				self.simulations = simulations_value(input_simulations)
			except ValueError:
				print("Invalid simulation count")

		input_attackers = input(f"Attackers [{self.attackers}]: ").strip()
		if input_attackers:
			try:
				self.attackers = int(input_attackers)
			except ValueError:
				print("Invalid attacker count")

		input_defenders = input(f"Defenders [{self.defenders}]: ").strip()
		if input_defenders:
			try:
				self.defenders = int(input_defenders)
			except ValueError:
				print("Invalid defender count")

		print(f"\nCurrent parameters: {simulations_label(self.simulations)} simulations, {self.attackers} attackers, {self.defenders} defenders")

	def run_sim(self, processor):
		task = job(processor, self.simulations, self.attackers, self.defenders)
		time, results = run_simulation(task)
		print_results(task, results)

	def list_sims(self):
		saves = list_saves()
		if not saves:
			print("\nNo saved simulations found")
			return

		print("\nSaved simulations:")
		for i, sim in enumerate(saves):
			slug = make_slug(sim)
			print(f"{i}: {slug}")

	def select_sims(self):
		saves = list_saves()
		if not saves:
			print("\nNo saved simulations found")
			return

		print("\nSaved simulations:")
		for i, sim in enumerate(saves):
			slug = make_slug(sim)
			marker = "*" if sim in self.selected else " "
			print(f"{marker} {i}: {slug}")

		indices_input = input("\nEnter comma-separated indices to add to selection: ").strip()
		if not indices_input:
			return

		try:
			indices = [int(x.strip()) for x in indices_input.split(',')]
			for idx in indices:
				if 0 <= idx < len(saves):
					if saves[idx] not in self.selected:
						self.selected.append(saves[idx])
				else:
					print(f"Invalid index: {idx}")
			print(f"\nSelected {len(self.selected)} simulation(s)")
		except ValueError:
			print("Invalid input format")

	def clear_selection(self):
		self.selected = []
		print("\nSelection cleared")

	def graph_selected(self):
		if not self.selected:
			print("\nNo simulations selected")
			return

		pyplot.figure(figsize=(10, 6))
		colors = ['#76B900', '#ED1C24', '#4A90E2', '#F7B731', '#5F27CD']

		for i, sim in enumerate(self.selected):
			slug = make_slug(sim)
			time, results = load_results(slug)
			color = colors[i % len(colors)]
			pyplot.bar(range(len(results)), results, alpha=0.5, label=slug, color=color)

		pyplot.xlabel("Attackers Lost")
		pyplot.ylabel("Frequency")
		pyplot.title("Risk Battle Simulation Results")
		pyplot.legend()
		pyplot.show()

	def print_selected(self):
		if not self.selected:
			print("\nNo simulations selected")
			return

		for sim in self.selected:
			slug = make_slug(sim)
			time, results = load_results(slug)
			print(f"\n{slug}:")
			print_results(sim, results)

	def run(self):
		commands = {
			'simulation': self.edit_params,
			'list': self.list_sims,
			'select': self.select_sims,
			'clear': self.clear_selection,
			'graph': self.graph_selected,
			'print': self.print_selected,
			'test': test_kernel_performance,
		}

		print("Risk Battle Simulator")
		print("Commands: simulation, run [cpu|cuda], list, select, clear, graph, print, exit")

		while True:
			try:
				cmd = input("\n> ").strip().lower()

				if not cmd:
					continue

				if cmd == 'exit':
					break

				if cmd.startswith('run '):
					processor_name = cmd[4:].strip()
					processor_map = {'cpu': Processor.CPU, 'cuda': Processor.GPU}
					processor = processor_map.get(processor_name)
					if processor:
						self.run_sim(processor)
					else:
						print("Invalid processor. Use 'cpu' or 'cuda'")
					continue

				handler = commands.get(cmd)
				if handler:
					handler()
				else:
					print("Unknown command. Try: simulation, run [cpu|cuda], list, select, clear, graph, print, exit")

			except KeyboardInterrupt:
				print("\n\nExiting...")
				break
			except Exception as e:
				print(f"Error: {e}")

if __name__ == "__main__":
	simulator = RiskSimulator()
	simulator.run()
