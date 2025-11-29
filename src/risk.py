import numpy
from matplotlib import pyplot
from utils.params import Processor, simulations_label, simulations_value, list_saves, print_results, run_simulation, job, make_slug, load_results
from procs.sim_gpu import test_kernel_performance
from procs.theoretical import from_start_losing_n

class RiskSimulator:
	def __init__(self):
		self.simulations = 1_000_000
		self.attackers = 75
		self.defenders = 10
		self.selected = {
			Processor.CPU: None,
			Processor.GPU: None,
			Processor.MATH: False,
		}

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
			is_selected = self.selected[sim.processor] == sim
			marker = "*" if is_selected else " "
			print(f"{marker} {i}: {slug}")

		print("\nCurrent selection:")
		print(f"  CPU: {make_slug(self.selected[Processor.CPU]) if self.selected[Processor.CPU] else 'None'}")
		print(f"  CUDA: {make_slug(self.selected[Processor.GPU]) if self.selected[Processor.GPU] else 'None'}")
		print(f"  Theoretical: {'Enabled' if self.selected[Processor.MATH] else 'Disabled'}")

		choice = input("\nSelect by index, 't' for theoretical, or Enter to cancel: ").strip().lower()

		if not choice:
			return

		if choice == 't':
			self.selected[Processor.MATH] = not self.selected[Processor.MATH]
			print(f"Theoretical {'enabled' if self.selected[Processor.MATH] else 'disabled'}")
			return

		try:
			idx = int(choice)
			if 0 <= idx < len(saves):
				sim = saves[idx]
				if self.selected[sim.processor] == sim:
					self.selected[sim.processor] = None
					print(f"Deselected {make_slug(sim)}")
				else:
					self.selected[sim.processor] = sim
					print(f"Selected {make_slug(sim)}")
			else:
				print(f"Invalid index: {idx}")
		except ValueError:
			print("Invalid input")

	def clear_selection(self):
		self.selected = {
			Processor.CPU: None,
			Processor.GPU: None,
			Processor.MATH: False,
		}
		print("\nSelection cleared")

	def graph_selected(self):
		if not self.selected:
			print("\nNo simulations selected")
			return

		print("\nGraph types:")
		print("1: Normalized probability")
		print("2: Cumulative distribution")
		print("3: Log scale probability")
		choice = input("Select graph type (1-3): ").strip()

		match choice:
			case '1':
				self._graph_normalized()
			case '2':
				self._graph_cumulative()
			case '3':
				self._graph_log_scale()
			case _:
				print("Invalid choice")

	def _prepare_data(self):
		data = []

		for processor in [Processor.CPU, Processor.GPU]:
			sim = self.selected[processor]
			if not sim:
				continue

			slug = make_slug(sim)
			time, results = load_results(slug)

			match processor:
				case Processor.CPU:
					color = '#0071C5'
					label = f"CPU ({simulations_label(sim.params.simulations)})"
				case Processor.GPU:
					color = '#76B900'
					label = f"CUDA ({simulations_label(sim.params.simulations)})"

			data.append({
				'results': results,
				'color': color, # type: ignore
				'label': label, # type: ignore
				'sim': sim
			})

		theoretical = None
		if self.selected[Processor.MATH]:
			theoretical = from_start_losing_n(self.attackers, self.defenders)

		return data, theoretical

	def _graph_normalized(self):
		data, theoretical = self._prepare_data()
		pyplot.figure(figsize=(12, 7))

		for d in data:
			normalized = d['results'] / numpy.sum(d['results'])
			pyplot.bar(range(len(normalized)), normalized, alpha=0.5,
				label=d['label'], color=d['color'])

		if theoretical:
			theoretical_floats = [float(p) for p in theoretical]
			pyplot.plot(range(len(theoretical_floats)), theoretical_floats,
				'o-', color='#9B59B6', label='Theoretical', linewidth=2, markersize=4)

		pyplot.xlabel("Attackers Lost")
		pyplot.ylabel("Probability")
		pyplot.title("Risk Battle Simulation - Normalized Probability")
		pyplot.legend()
		pyplot.grid(True, alpha=0.3)
		pyplot.show()

	def _graph_cumulative(self):
		data, theoretical = self._prepare_data()
		pyplot.figure(figsize=(12, 7))

		for d in data:
			normalized = d['results'] / numpy.sum(d['results'])
			cumulative = numpy.cumsum(normalized)
			pyplot.plot(range(len(cumulative)), cumulative,
				label=d['label'], color=d['color'], linewidth=2)

		if theoretical:
			theoretical_floats = [float(p) for p in theoretical]
			cumulative_theoretical = numpy.cumsum(theoretical_floats)
			pyplot.plot(range(len(cumulative_theoretical)), cumulative_theoretical,
				'--', color='#9B59B6', label='Theoretical', linewidth=2)

		pyplot.xlabel("Attackers Lost")
		pyplot.ylabel("Cumulative Probability")
		pyplot.title("Risk Battle Simulation - Cumulative Distribution")
		pyplot.legend()
		pyplot.grid(True, alpha=0.3)
		pyplot.show()

	def _graph_log_scale(self):
		data, theoretical = self._prepare_data()
		pyplot.figure(figsize=(12, 7))

		for d in data:
			normalized = d['results'] / numpy.sum(d['results'])
			non_zero = normalized > 0
			x_vals = numpy.where(non_zero)[0]
			y_vals = normalized[non_zero]
			pyplot.semilogy(x_vals, y_vals, 'x',
				label=d['label'], color=d['color'], markersize=8)

		if theoretical:
			theoretical_floats = [float(p) for p in theoretical]
			non_zero_indices = [i for i, p in enumerate(theoretical_floats) if p > 0]
			non_zero_probs = [theoretical_floats[i] for i in non_zero_indices]
			pyplot.semilogy(non_zero_indices, non_zero_probs,
				'o-', color='#9B59B6', label='Theoretical', linewidth=2, markersize=4)

		pyplot.xlabel("Attackers Lost")
		pyplot.ylabel("Probability (log scale)")
		pyplot.title("Risk Battle Simulation - Log Scale Probability")
		pyplot.legend()
		pyplot.grid(True, alpha=0.3, which='both')
		pyplot.show()

	def print_selected(self):
		if not any([self.selected[Processor.CPU], self.selected[Processor.GPU], self.selected[Processor.MATH]]):
			print("\nNo simulations selected")
			return

		for processor in [Processor.CPU, Processor.GPU]:
			sim = self.selected[processor]
			if sim:
				slug = make_slug(sim)
				time, results = load_results(slug)
				print(f"\n{slug}:")
				print_results(sim, results)

		if self.selected[Processor.MATH]:
			theoretical = from_start_losing_n(self.attackers, self.defenders)
			print(f"\nTheoretical (A{self.attackers}-D{self.defenders}):")
			for i, prob in enumerate(theoretical):
				print(f"{i:>2} attackers lost {float(prob):.15e}", end='  |  ' if (i + 1) % 5 else '\r\n')

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
