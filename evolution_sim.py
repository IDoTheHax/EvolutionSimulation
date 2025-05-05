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
        self.mutation_rate = 0.08
        self.environment_pressure = 2.5  # Increased from 1.8
        self.max_fitness = 1.0
        self.min_population_size = 10
        self.carrying_capacity = 50000
        self.running = False  # Flag to control simulation running
        self.max_infection_radius = 40  # Maximum distance for disease spread
        self.max_render_individuals = 5000  # Limit number of individuals rendered on the canvas
        self.disease_mortality = 0.4    # New parameter for disease impact
        self.reproduction_rate = 1.2    # New parameter - lower than previous 2.0
        self.minimum_fitness = 0.3      # New parameter for survival threshold
        

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
        
        # Initialize all population attributes with the same size
        self.fitness = torch.rand(self.population_size, device=self.device) * 0.6 + 0.2
        self.immunity = torch.zeros(self.population_size, device=self.device)
        self.genetic_diversity = torch.rand(self.population_size, device=self.device)
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
        """Spread disease based on population density and fitness."""
        if not self.infected.any():
            return
    
        # Ensure all tensors have the same size
        current_size = len(self.positions)
        self.fitness = self.fitness[:current_size]
        self.immunity = self.immunity[:current_size]
        self.infected = self.infected[:current_size]
        self.genetic_diversity = self.genetic_diversity[:current_size]
    
        infected_indices = torch.where(self.infected)[0]
        
        for idx in infected_indices:
            if idx >= current_size:  # Safety check
                continue
                
            x, y = self.positions[idx].tolist()
            distances = torch.norm(self.positions.float() - torch.tensor([x, y], device=self.device).float(), dim=1)
            
            # Calculate local population density
            local_density = torch.sum(distances < self.max_infection_radius).float() / (3.14 * self.max_infection_radius ** 2)
            
            # Infection chance based on density, fitness, and immunity
            within_radius = distances < self.max_infection_radius
            base_infection_chance = 0.1 * local_density * self.environment_pressure
            
            # Create infection chance tensor
            infection_chance = torch.zeros(current_size, device=self.device)
            valid_targets = within_radius & ~self.infected
            
            if valid_targets.any():
                infection_chance[valid_targets] = (
                    base_infection_chance *
                    (1 - self.fitness[valid_targets]) *
                    (1 - self.immunity[valid_targets])
                )
            
            # Apply infections
            new_infections = torch.where(torch.rand(current_size, device=self.device) < infection_chance)[0]
            self.infected[new_infections] = True
            
            # Increase immunity for survivors
            self.immunity[new_infections] += 0.1
            self.immunity.clamp_(0, 1)

    
    def remove_weak_individuals(self):
        """Remove individuals based on fitness and disease status."""
        current_size = len(self.positions)
        
        # Ensure all tensors have the same size
        self.fitness = self.fitness[:current_size]
        self.immunity = self.immunity[:current_size]
        self.infected = self.infected[:current_size]
        self.genetic_diversity = self.genetic_diversity[:current_size]
        
        # Calculate survival probability
        survival_chance = self.fitness.clone()
        
        # Stronger disease impact
        infected_mask = self.infected
        survival_chance[infected_mask] *= (1 - self.disease_mortality * (1 - self.immunity[infected_mask]))
        
        # Environmental pressure based on population density
        population_pressure = current_size / self.carrying_capacity
        survival_threshold = self.minimum_fitness + (population_pressure * 0.3)
        
        # Add random variation to threshold
        random_factor = torch.rand(current_size, device=self.device) * 0.2
        survival_threshold = torch.ones_like(survival_chance) * survival_threshold + random_factor
        
        # Determine survivors - stricter conditions
        survivors = (survival_chance > survival_threshold) & (self.fitness > self.minimum_fitness)
        
        # Update population
        self.positions = self.positions[survivors]
        self.fitness = self.fitness[survivors]
        self.infected = self.infected[survivors]
        self.immunity = self.immunity[survivors]
        self.genetic_diversity = self.genetic_diversity[survivors]

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
        """Reproduce population with genetic inheritance."""
        current_size = len(self.positions)
        
        # Calculate target population based on carrying capacity
        target_population = self.carrying_capacity * 0.8  # Aim for 80% of carrying capacity
        
        if current_size < target_population:
            # Adjust reproduction rate based on population density
            density_factor = 1 - (current_size / self.carrying_capacity)
            adjusted_rate = self.reproduction_rate * density_factor
            
            # Calculate number of offspring
            num_offspring = min(
                int(current_size * adjusted_rate),
                self.carrying_capacity - current_size
            )
            
            if num_offspring <= 0:
                return
                
            # Select parents based on fitness
            fitness_weights = self.fitness / self.fitness.sum()
            parent_indices = torch.multinomial(fitness_weights, 
                                             num_offspring, 
                                             replacement=True)
            
            # Generate offspring positions
            offspring_positions = self.positions[parent_indices]
            
            # Inherit fitness with mutation
            parent_fitness = self.fitness[parent_indices]
            mutation = torch.randn(num_offspring, device=self.device) * self.mutation_rate
            offspring_fitness = parent_fitness + mutation
            offspring_fitness.clamp_(0, self.max_fitness)
            
            # Inherit immunity with decay
            offspring_immunity = self.immunity[parent_indices] * 0.8
            
            # Generate new genetic diversity
            offspring_diversity = (self.genetic_diversity[parent_indices] + 
                                 torch.rand(num_offspring, device=self.device)) / 2
            
            # Add offspring to population
            self.positions = torch.cat((self.positions, offspring_positions))
            self.fitness = torch.cat((self.fitness, offspring_fitness))
            self.immunity = torch.cat((self.immunity, offspring_immunity))
            self.genetic_diversity = torch.cat((self.genetic_diversity, offspring_diversity))
            self.infected = torch.cat((
                self.infected,
                torch.zeros(num_offspring, dtype=torch.bool, device=self.device)
            ))
    
    def handle_disease_extinction(self):
        """Handle disease extinction based on population immunity."""
        if not self.infected.any():
            return
            
        # Calculate average immunity of infected population
        infected_immunity = self.immunity[self.infected].mean().item()
        infected_fitness = self.fitness[self.infected].mean().item()
        
        # Disease extinction chance increases with immunity and fitness
        extinction_chance = (infected_immunity * 0.5 + infected_fitness * 0.5) ** 2
        
        if random.random() < extinction_chance:
            self.infected[:] = False
            print(f"Disease extinct at generation {self.generation}")

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