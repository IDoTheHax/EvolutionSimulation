import tkinter as tk
import torch  # Use PyTorch with ROCm for GPU acceleration
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys


class EvolutionSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Mass Extinction Test (Predators Spread Disease)")
        self.canvas = tk.Canvas(master, width=800, height=600, bg='white')
        self.canvas.pack()

        # Simulation parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use ROCm if available
        self.initial_population_size = 1200
        self.population_size = self.initial_population_size
        self.predator_count = 10
        self.predator_targets_per_generation = 1
        self.generation = 0
        self.mutation_rate = 0.05
        self.environment_pressure = 1.8
        self.max_fitness = 1.0
        self.min_population_size = 10
        self.carrying_capacity = 100000
        self.running = False  # Flag to control simulation running
        self.max_infection_radius = 40  # Maximum distance for disease spread
        self.max_render_individuals = 5000  # Limit number of individuals rendered on the canvas

        # Population data stored as PyTorch tensors
        self.positions = torch.zeros((self.population_size, 2), device=self.device)  # x, y positions
        self.fitness = torch.rand(self.population_size, device=self.device) * 0.3 + 0.4  # Fitness (0.4 to 0.7)
        self.infected = torch.zeros(self.population_size, dtype=torch.bool, device=self.device)  # Disease state (True for infected)

        # Predator infection status
        self.predator_infected = [False] * self.predator_count  # Track if predators are spreading the disease

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

        # Stats panel
        self.stats_frame = tk.Frame(master)
        self.stats_frame.pack()
        self.generation_label = tk.Label(self.stats_frame, text="Generation: 0")
        self.generation_label.pack(side=tk.LEFT)
        self.average_fitness_label = tk.Label(self.stats_frame, text="Average Fitness: 0.0")
        self.average_fitness_label.pack(side=tk.LEFT, padx=10)
        self.population_size_label = tk.Label(self.stats_frame, text="Population Size: 0")
        self.population_size_label.pack(side=tk.LEFT, padx=10)

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
        """Initialize the population with random positions and fitness."""
        x = torch.randint(50, 750, (self.population_size,), device=self.device)
        y = torch.randint(50, 550, (self.population_size,), device=self.device)
        self.positions = torch.stack((x, y), dim=-1)
        self.fitness = torch.rand(self.population_size, device=self.device) * 0.3 + 0.4
        self.infected = torch.zeros(self.population_size, dtype=torch.bool, device=self.device)

        # Start with one infected individual
        patient_zero = random.randint(0, self.population_size - 1)
        self.infected[patient_zero] = True

    def start_simulation(self):
        """Start the simulation."""
        self.running = True
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

        # Apply disease spread visually
        self.spread_disease()

        # Remove individuals with zero fitness
        self.remove_weak_individuals()

        # Predators target the weakest individuals and spread disease
        self.apply_predation()

        # Reproduce if population is below carrying capacity
        self.reproduce_population()

        # Handle disease extinction chance
        self.handle_disease_extinction()

        # Update stats and render individuals
        self.update_stats_and_render()

        # Schedule next generation
        if self.running and self.generation < 250000:
            self.master.after(50, self.simulate_generation)

    def spread_disease(self):
        """Spread disease visually based on proximity and fitness."""
        infected_indices = torch.where(self.infected)[0]
        for idx in infected_indices:
            x, y = self.positions[idx].tolist()
            distances = torch.norm(self.positions.float() - torch.tensor([x, y], device=self.device).float(), dim=1)

            # Limit spread to within the maximum infection radius
            within_radius = distances < self.max_infection_radius
            infection_chance = (1 - self.fitness) * self.environment_pressure * 0.05 / (distances + 1)
            infection_chance[~within_radius] = 0  # No infection beyond the radius

            new_infections = torch.where(torch.rand(self.positions.size(0), device=self.device) < infection_chance)[0]
            self.infected[new_infections] = True

    def remove_weak_individuals(self):
        """Remove individuals with zero fitness."""
        self.fitness[self.infected] -= 0.1  # Fitness penalty for infected individuals
        self.fitness = torch.clamp(self.fitness, 0, self.max_fitness)
        alive = self.fitness > 0
        self.positions = self.positions[alive]
        self.fitness = self.fitness[alive]
        self.infected = self.infected[alive]

    def apply_predation(self):
        """Predators target the weakest individuals and spread disease."""
        for predator_idx in range(self.predator_count):
            if not len(self.positions):
                break

            # Target the weakest individual
            weakest_index = torch.argmin(self.fitness)

            # Update predator's infection state
            if self.infected[weakest_index]:
                self.predator_infected[predator_idx] = True

            # Spread disease if the predator is infected
            if self.predator_infected[predator_idx]:
                self.infected[weakest_index] = True

            # Remove the individual from the population
            self.positions = torch.cat((self.positions[:weakest_index], self.positions[weakest_index + 1:]))
            self.fitness = torch.cat((self.fitness[:weakest_index], self.fitness[weakest_index + 1:]))
            self.infected = torch.cat((self.infected[:weakest_index], self.infected[weakest_index + 1:]))

    def reproduce_population(self):
        """Reproduce individuals to maintain the population size."""
        if len(self.fitness) < self.carrying_capacity:
            num_offspring = len(self.fitness) * 2
            num_offspring = min(num_offspring, self.carrying_capacity - len(self.fitness))
            offspring_positions = self.positions[
                torch.randint(0, len(self.positions), (num_offspring,), device=self.device)
            ]
            offspring_fitness = self.fitness[
                torch.randint(0, len(self.fitness), (num_offspring,), device=self.device)
            ] + torch.rand(num_offspring, device=self.device) * 0.2 - 0.1
            offspring_fitness = torch.clamp(offspring_fitness, 0, self.max_fitness)

            # Append offspring to the population
            self.positions = torch.cat((self.positions, offspring_positions), dim=0)
            self.fitness = torch.cat((self.fitness, offspring_fitness), dim=0)
            self.infected = torch.cat((self.infected, torch.zeros(num_offspring, dtype=torch.bool, device=self.device)))

    def handle_disease_extinction(self):
        """Handle the chance for the disease to go extinct."""
        if not self.infected.any():
            return
        infected_fitness = self.fitness[self.infected]
        avg_infected_fitness = infected_fitness.mean().item()
        if avg_infected_fitness > 0.7:  # High average fitness reduces disease survival
            self.infected[:] = False

    def update_stats_and_render(self):
        """Update stats and render individuals."""
        avg_fitness = self.fitness.mean().item()
        pop_size = len(self.fitness)
        self.average_fitness_label.config(text=f"Average Fitness: {avg_fitness:.3f}")
        self.population_size_label.config(text=f"Population Size: {pop_size}")

        # Render population (limit to max_render_individuals for performance)
        self.canvas.delete("all")
        render_count = min(self.max_render_individuals, len(self.positions))
        for i in range(render_count):
            x, y = self.positions[i].tolist()
            if self.infected[i]:
                color = "#FF00FF"  # Magenta for infected
            else:
                color = "#%02x%02x%02x" % (
                    int(self.fitness[i].item() * 255),
                    0,
                    255 - int(self.fitness[i].item() * 255),
                )
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")

        # Collect data
        self.generation_data.append(self.generation)
        self.fitness_data.append(avg_fitness)
        self.population_data.append(pop_size)

        # Update graph
        self.ax.clear()
        self.ax.plot(self.generation_data, self.fitness_data, label="Average Fitness", color="blue")
        self.ax.plot(self.generation_data, self.population_data, label="Population Size", color="green")
        self.ax.set_title("Population and Fitness Trends")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Values")
        self.ax.legend()
        self.graph_canvas.draw()

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
        """Handle the window close event."""
        self.running = False
        self.master.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    sim = EvolutionSimulation(root)
    root.mainloop()