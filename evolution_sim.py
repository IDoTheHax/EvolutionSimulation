import tkinter as tk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statistics
import sys


class EvolutionSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("1000 circles vs 1 deadly disease")
        self.canvas = tk.Canvas(master, width=800, height=600, bg='white')
        self.canvas.pack()

        # Simulation parameters
        self.initial_population_size = 800
        self.population_size = self.initial_population_size
        self.predator_count = 10
        self.predator_targets_per_generation = 1
        self.generation = 0
        self.individuals = []
        self.predators = []
        self.mutation_rate = 0.05
        self.environment_pressure = 2.5
        self.max_fitness = 1.0
        self.min_population_size = 10
        self.carrying_capacity = 15000
        self.running = False  # Flag to control simulation running

        # Data collection
        self.fitness_data = []
        self.population_data = []
        self.generation_data = []

        self.create_population()

        # Control panel
        self.control_frame = tk.Frame(master)
        self.control_frame.pack()
        self.start_button = tk.Button(self.control_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(self.control_frame, text="Stop Simulation", command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT)
        self.reset_button = tk.Button(self.control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT)

        # Scenario selector
        self.scenario_label = tk.Label(self.control_frame, text="Scenario:")
        self.scenario_label.pack(side=tk.LEFT)
        self.scenario_var = tk.StringVar(value="stable")
        self.scenario_menu = tk.OptionMenu(self.control_frame, self.scenario_var, "stable", "island", "environmental_change")
        self.scenario_menu.pack(side=tk.LEFT)

        # Stats panel
        self.stats_frame = tk.Frame(master)
        self.stats_frame.pack()
        self.generation_label = tk.Label(self.stats_frame, text="Generation: 0")
        self.generation_label.pack(side=tk.LEFT)

        # Metrics display
        self.metrics_frame = tk.Frame(master)
        self.metrics_frame.pack()
        self.average_fitness_label = tk.Label(self.metrics_frame, text="Average Fitness: 0.0")
        self.average_fitness_label.pack(side=tk.LEFT, padx=10)
        self.population_size_label = tk.Label(self.metrics_frame, text="Population Size: 0")
        self.population_size_label.pack(side=tk.LEFT, padx=10)
        self.environment_pressure_label = tk.Label(self.metrics_frame, text="Environmental Pressure: 2.5")
        self.environment_pressure_label.pack(side=tk.LEFT, padx=10)
        self.fitness_stddev_label = tk.Label(self.metrics_frame, text="Fitness StdDev: 0.0")
        self.fitness_stddev_label.pack(side=tk.LEFT, padx=10)

        # Fitness and population graph
        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("Population and Fitness Trends")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Values")
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=self.stats_frame)
        self.graph_canvas.get_tk_widget().pack()

        # Handle window close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_population(self):
        """Create initial population with genetic variation."""
        self.individuals = []
        for _ in range(self.population_size):
            x, y = random.randint(50, 750), random.randint(50, 550)
            fitness = random.uniform(0.4, 0.7)  # Moderate starting fitness
            color = self.fitness_to_color(fitness)
            ind = {"x": x, "y": y, "fitness": fitness, "color": color}
            self.individuals.append(ind)
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline="")

    def create_predators(self):
        """Create predators for the simulation."""
        self.predators = []
        for _ in range(self.predator_count):
            x, y = random.randint(50, 750), random.randint(50, 550)
            predator = {"x": x, "y": y}
            self.predators.append(predator)
            self.canvas.create_rectangle(x-5, y-5, x+5, y+5, fill="red", outline="")

    def fitness_to_color(self, fitness):
        """Convert fitness to a color for visualization."""
        return "#%02x%02x%02x" % (int(fitness * 255), 0, 255 - int(fitness * 255))

    def start_simulation(self):
        """Start the simulation."""
        self.running = True
        self.scenario = self.scenario_var.get()
        self.simulate_generation()

    def stop_simulation(self):
        """Stop the simulation."""
        self.running = False

    def simulate_generation(self):
        """Simulate one generation of evolution."""
        if not self.running:
            return

        self.generation += 1
        self.generation_label.config(text=f"Generation: {self.generation}")
        self.canvas.delete("all")

        # Debugging: Print current population size
        print(f"Generation {self.generation}: Current population size = {len(self.individuals)}")

        # Create predators
        self.create_predators()

        # Apply predation
        for predator in self.predators:
            for _ in range(self.predator_targets_per_generation):
                if not self.individuals:  # Stop if the population is extinct
                    break
                target = random.choices(
                    self.individuals,
                    weights=[1 - ind["fitness"] for ind in self.individuals],  # Lower fitness = higher chance of being targeted
                    k=1
                )[0]
                self.individuals.remove(target)

        # Check for extinction
        if not self.individuals:
            print("Population extinct! Repopulating with random individuals...")
            self.individuals = [
                {
                    "x": random.randint(50, 750),
                    "y": random.randint(50, 550),
                    "fitness": random.uniform(0.4, 0.7),
                    "color": self.fitness_to_color(random.uniform(0.4, 0.7)),
                }
                for _ in range(self.min_population_size)
            ]

        # Apply disease
        self.apply_disease()

        # Prevent population from dropping below minimum size
        if len(self.individuals) < self.min_population_size:
            print(f"Warning: Population size too low ({len(self.individuals)}). Repopulating to minimum size...")
            self.individuals += [
                {
                    "x": random.randint(50, 750),
                    "y": random.randint(50, 550),
                    "fitness": random.uniform(0.4, 0.7),
                    "color": self.fitness_to_color(random.uniform(0.4, 0.7)),
                }
                for _ in range(self.min_population_size - len(self.individuals))
            ]

        # Reproduce to fill population dynamically
        survivors = self.individuals
        new_population = []
        while len(new_population) < len(survivors) * random.uniform(1.2, 1.5):  # Add randomness to reproduction rate
            if len(new_population) >= self.carrying_capacity:  # Stop at carrying capacity
                break
            parent1, parent2 = random.sample(survivors, 2)
            child_fitness = (parent1["fitness"] + parent2["fitness"]) / 2
            # Mutation
            if random.uniform(0, 1) < self.mutation_rate:
                mutation_effect = random.uniform(-0.1, 0.1)
                child_fitness += mutation_effect
            child_fitness = max(0, min(self.max_fitness, child_fitness))  # Cap fitness
            x, y = random.randint(50, 750), random.randint(50, 550)
            color = self.fitness_to_color(child_fitness)
            new_population.append({"x": x, "y": y, "fitness": child_fitness, "color": color})

        # Update individuals
        self.individuals = new_population

        # Debugging: Check population size after reproduction
        print(f"Generation {self.generation}: Population size after reproduction = {len(self.individuals)}")

        # Redraw population
        for ind in self.individuals:
            self.canvas.create_oval(ind["x"]-5, ind["y"]-5, ind["x"]+5, ind["y"]+5, fill=ind["color"], outline="")

        # Collect and plot data
        if len(self.individuals) > 0:  # Avoid division by zero
            avg_fitness = sum(ind["fitness"] for ind in self.individuals) / len(self.individuals)
            fitness_stddev = statistics.stdev(ind["fitness"] for ind in self.individuals) if len(self.individuals) > 1 else 0
        else:
            avg_fitness = 0
            fitness_stddev = 0

        self.generation_data.append(self.generation)
        self.fitness_data.append(avg_fitness)
        self.population_data.append(len(self.individuals))

        self.ax.clear()
        self.ax.plot(self.generation_data, self.fitness_data, label="Average Fitness", color="blue")
        self.ax.plot(self.generation_data, self.population_data, label="Population Size", color="green")
        self.ax.set_title("Population and Fitness Trends")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Values")
        self.ax.legend()
        self.graph_canvas.draw()

        # Schedule next generation
        if self.running and self.generation < 1000:  # Run for 1000 generations
            self.master.after(50, self.simulate_generation)

    def apply_disease(self):
        """Simulate disease affecting the population."""
        infected = []
        for individual in self.individuals:
            # Infection chance increases with environmental pressure and lower fitness
            infection_chance = (1 - individual["fitness"]) * self.environment_pressure * 0.1
            if random.uniform(0, 1) < infection_chance:
                infected.append(individual)

        # Debugging: Log number of infected individuals
        print(f"Generation {self.generation}: Infected individuals = {len(infected)}")

        # Apply effects of disease
        for individual in infected:
            individual["fitness"] -= random.uniform(0.1, 0.3)  # Reduce fitness
            individual["fitness"] = max(0, individual["fitness"])  # Cap fitness at 0
            individual["color"] = "#FF00FF"  # Change color to magenta to indicate infection

        # Remove individuals with zero fitness
        self.individuals = [ind for ind in self.individuals if ind["fitness"] > 0]

    def reset_simulation(self):
        """Reset the simulation."""
        self.running = False
        self.generation = 0
        self.generation_label.config(text="Generation: 0")
        self.fitness_data = []
        self.population_data = []
        self.generation_data = []
        self.ax.clear()
        self.graph_canvas.draw()
        self.create_population()

    def on_close(self):
        """Handle window close event."""
        self.running = False
        self.master.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    sim = EvolutionSimulation(root)
    root.mainloop()