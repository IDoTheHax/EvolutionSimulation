import tkinter as tk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EvolutionSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Please Work Evolution Simulation")
        self.canvas = tk.Canvas(master, width=800, height=600, bg='white')
        self.canvas.pack()

        # Simulation parameters
        self.population_size = 100
        self.generation = 0
        self.individuals = []
        self.mutation_rate = 0.05
        self.environment_pressure = 0.5
        self.max_fitness = 1.0
        self.min_population_size = 10
        self.carrying_capacity = 150
        self.scenario = "stable"  # Options: stable, island, environmental_change

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

        # Fitness and population graph
        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("Population and Fitness Trends")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Values")
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=self.stats_frame)
        self.graph_canvas.get_tk_widget().pack()

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

    def fitness_to_color(self, fitness):
        """Convert fitness to a color for visualization."""
        return "#%02x%02x%02x" % (int(fitness * 255), 0, 255 - int(fitness * 255))

    def start_simulation(self):
        """Run the simulation."""
        self.scenario = self.scenario_var.get()
        self.simulate_generation()

    def simulate_generation(self):
        """Simulate one generation of evolution."""
        self.generation += 1
        self.generation_label.config(text=f"Generation: {self.generation}")
        self.canvas.delete("all")

        # Apply natural selection
        total_fitness = sum(ind["fitness"] for ind in self.individuals)
        if total_fitness == 0:  # Prevent division by zero
            total_fitness = 1
        survivors = [
            ind for ind in self.individuals
            if random.uniform(0, 1) < (ind["fitness"] / total_fitness) * self.environment_pressure
        ]

        # Prevent extinction
        if len(survivors) < self.min_population_size:
            survivors += random.choices(self.individuals, k=self.min_population_size - len(survivors))

        # Reproduce to fill population
        new_population = []
        while len(new_population) < self.population_size and len(new_population) < self.carrying_capacity:
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

        # Redraw population
        for ind in self.individuals:
            self.canvas.create_oval(ind["x"]-5, ind["y"]-5, ind["x"]+5, ind["y"]+5, fill=ind["color"], outline="")

        # Collect and plot data
        avg_fitness = sum(ind["fitness"] for ind in self.individuals) / len(self.individuals)
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

        # Update environmental pressure dynamically (for environmental_change scenario)
        if self.scenario == "environmental_change":
            self.environment_pressure = max(0.2, min(1.0, self.environment_pressure + random.uniform(-0.02, 0.02)))

        # Schedule next generation
        if self.generation < 100:  # Run for 100 generations
            self.master.after(500, self.simulate_generation)

    def reset_simulation(self):
        """Reset the simulation."""
        self.generation = 0
        self.generation_label.config(text="Generation: 0")
        self.fitness_data = []
        self.population_data = []
        self.generation_data = []
        self.ax.clear()
        self.graph_canvas.draw()
        self.create_population()

if __name__ == "__main__":
    root = tk.Tk()
    sim = EvolutionSimulation(root)
    root.mainloop()