import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


class GameParameters:
    """Class containing the parameters of the discrete-continuous game."""

    def __init__(self, n_attacks: int = None, m_defenses: int = None, seed: int = None):
        """
        Initialize game parameters for the DDoS attack defense scenario.

        Args:
            n_attacks: Number of possible attack vectors
            m_defenses: Number of possible defense mechanisms
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.n = n_attacks  # Number of possible attack vectors
        self.m = m_defenses  # Number of possible defense mechanisms

        # Define indices sets
        self.N = list(range(self.n))
        self.M = list(range(self.m))

        # Initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize all game parameters with realistic values for DDoS scenario."""
        # Potential damage from each attack type (in monetary value or service disruption score)
        self.u = np.random.uniform(10000, 100000, self.n)  # Damage values for each attack

        # Attack costs - different types of DDoS attacks have different implementation costs
        self.c_attack = np.random.uniform(1000, 10000, self.n)

        # Defense costs - firewalls, rate limiters, traffic analyzers, etc.
        self.c_defense = np.random.uniform(5000, 50000, self.m)

        # Maximum investment limits
        self.w_max = np.random.uniform(1000, 5000, self.n)
        self.z_max = np.random.uniform(5000, 20000, self.m)

        # Budget constraints
        self.C_max_defense = np.sum(self.c_defense) * 0.6  # 60% of total possible cost
        self.C_max_attack = np.sum(self.c_attack) * 0.7  # 70% of total possible cost

        # Base effectiveness of defenses against attacks (probability matrix)
        self.P = np.random.uniform(0.1, 0.9, (self.n, self.m))

        # Investment efficiency coefficients
        self.alpha_defense = np.random.uniform(0.001, 0.005, self.m)  # Diminishing returns
        self.alpha_attack = np.random.uniform(0.001, 0.003, self.n)  # Diminishing returns

        # Time period
        self.T = 1.0  # Normalized time period

        # Print key parameters
        print(f"Defense budget: ${self.C_max_defense:.2f}")
        print(f"Attack budget: ${self.C_max_attack:.2f}")

    def calculate_defense_effectiveness(self, z: np.ndarray) -> np.ndarray:
        """Calculate defense effectiveness boost from investments."""
        effectiveness = np.zeros(self.m)
        for j in range(self.m):
            effectiveness[j] = 1.0 - np.exp(-self.alpha_defense[j] * z[j])
        return effectiveness

    def calculate_attack_effectiveness(self, w: np.ndarray) -> np.ndarray:
        """Calculate attack effectiveness boost from investments."""
        effectiveness = np.zeros(self.n)
        for i in range(self.n):
            effectiveness[i] = 1.0 - np.exp(-self.alpha_attack[i] * w[i])
        return effectiveness


@dataclass
class DefenseStrategy:
    """Represents a defense strategy with discrete and continuous components."""
    x: np.ndarray  # Binary vector for selecting defenses
    z: np.ndarray  # Continuous vector for investment levels
    fitness: float = float('-inf')  # Initialize with worst possible fitness


@dataclass
class AttackStrategy:
    """Represents an attack strategy with discrete and continuous components."""
    y: np.ndarray  # Binary vector for selecting attacks
    w: np.ndarray  # Continuous vector for investment levels
    fitness: float = float('-inf')  # Initialize with worst possible fitness


class DiscreteContGameSolver:
    """Solver for the discrete-continuous game using coevolutionary genetic algorithm."""

    def __init__(self, params: GameParameters, pop_size: int = 50,
                 max_gen: int = 100, crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2, tournament_size: int = 3):
        """
        Initialize the solver.

        Args:
            params: Game parameters
            pop_size: Population size for genetic algorithm
            max_gen: Maximum number of generations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            tournament_size: Tournament size for selection
        """
        self.params = params
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

        # Initialize populations
        self.defense_pop = self.initialize_defense_population()
        self.attack_pop = self.initialize_attack_population()

        # History tracking
        self.history = {
            'defense_best_fitness': [],
            'attack_best_fitness': [],
            'damage_values': [],
            'defense_choices': [],
            'attack_choices': []
        }

    def initialize_defense_population(self) -> List[DefenseStrategy]:
        """Initialize population of defense strategies."""
        population = []

        # Create diverse initial population
        for _ in range(self.pop_size):
            # Generate random binary selection vector
            x = np.random.randint(0, 2, self.params.m).astype(np.int8)

            # Generate random investment levels
            z = np.zeros(self.params.m)
            for j in range(self.params.m):
                if x[j] == 1:
                    z[j] = np.random.uniform(0, self.params.z_max[j])

            # Create strategy
            strategy = DefenseStrategy(x=x, z=z)

            # Ensure budget constraints
            self.repair_defense_strategy(strategy)

            population.append(strategy)

        # Add specialized strategies
        # 1. Cost-efficient strategy
        efficiency = self.params.P.sum(axis=0) / self.params.c_defense
        top_indices = np.argsort(efficiency)[::-1][:self.params.m // 3]
        x = np.zeros(self.params.m, dtype=np.int8)
        x[top_indices] = 1
        z = np.zeros(self.params.m)
        for j in top_indices:
            z[j] = np.random.uniform(0, self.params.z_max[j])
        strategy = DefenseStrategy(x=x, z=z)
        self.repair_defense_strategy(strategy)
        population.append(strategy)

        # 2. Coverage-maximizing strategy
        coverage = self.params.P.sum(axis=0)
        top_indices = np.argsort(coverage)[::-1][:self.params.m // 3]
        x = np.zeros(self.params.m, dtype=np.int8)
        x[top_indices] = 1
        z = np.zeros(self.params.m)
        for j in top_indices:
            z[j] = np.random.uniform(0, self.params.z_max[j])
        strategy = DefenseStrategy(x=x, z=z)
        self.repair_defense_strategy(strategy)
        population.append(strategy)

        return population

    def initialize_attack_population(self) -> List[AttackStrategy]:
        """Initialize population of attack strategies."""
        population = []

        # Create diverse initial population
        for _ in range(self.pop_size):
            # Generate random binary selection vector
            y = np.random.randint(0, 2, self.params.n).astype(np.int8)

            # Generate random investment levels
            w = np.zeros(self.params.n)
            for i in range(self.params.n):
                if y[i] == 1:
                    w[i] = np.random.uniform(0, self.params.w_max[i])

            # Create strategy
            strategy = AttackStrategy(y=y, w=w)

            # Ensure budget constraints
            self.repair_attack_strategy(strategy)

            population.append(strategy)

        # Add specialized strategies
        # 1. High-damage strategy
        top_indices = np.argsort(self.params.u)[::-1][:self.params.n // 3]
        y = np.zeros(self.params.n, dtype=np.int8)
        y[top_indices] = 1
        w = np.zeros(self.params.n)
        for i in top_indices:
            w[i] = np.random.uniform(0, self.params.w_max[i])
        strategy = AttackStrategy(y=y, w=w)
        self.repair_attack_strategy(strategy)
        population.append(strategy)

        # 2. Cost-efficient attack strategy
        efficiency = self.params.u / self.params.c_attack
        top_indices = np.argsort(efficiency)[::-1][:self.params.n // 3]
        y = np.zeros(self.params.n, dtype=np.int8)
        y[top_indices] = 1
        w = np.zeros(self.params.n)
        for i in top_indices:
            w[i] = np.random.uniform(0, self.params.w_max[i])
        strategy = AttackStrategy(y=y, w=w)
        self.repair_attack_strategy(strategy)
        population.append(strategy)

        return population

    def evaluate_defense_strategy(self, defense: DefenseStrategy, attack: AttackStrategy) -> float:
        """Evaluate fitness of a defense strategy against a given attack strategy."""
        x, z = defense.x, defense.z
        y, w = attack.y, attack.w

        # Calculate effectiveness boosts
        defense_effect = self.params.calculate_defense_effectiveness(z)
        attack_effect = self.params.calculate_attack_effectiveness(w)

        # Calculate modified prevention probabilities
        P_mod = np.zeros((self.params.n, self.params.m))
        for i in range(self.params.n):
            for j in range(self.params.m):
                if x[j] == 1 and y[i] == 1:
                    effectiveness = self.params.P[i, j] * (1 + defense_effect[j]) / (1 + attack_effect[i])
                    P_mod[i, j] = min(max(effectiveness, 0), 1)  # Ensure value is in [0, 1]

        # Calculate maximum damage (without defenses)
        max_damage = np.sum(self.params.u * y)

        # Calculate prevented damage
        prevented_damage = 0
        for i in range(self.params.n):
            if y[i] == 1:
                # Find most effective defense for this attack
                best_prevention = 0
                for j in range(self.params.m):
                    if x[j] == 1:
                        best_prevention = max(best_prevention, P_mod[i, j])

                prevented_damage += self.params.u[i] * best_prevention

        # Calculate residual damage
        residual_damage = max_damage - prevented_damage

        # Fitness is negative damage (we want to minimize damage)
        fitness = -residual_damage

        return fitness

    def evaluate_attack_strategy(self, attack: AttackStrategy, defense: DefenseStrategy) -> float:
        """Evaluate fitness of an attack strategy against a given defense strategy."""
        x, z = defense.x, defense.z
        y, w = attack.y, attack.w

        # Calculate effectiveness boosts
        defense_effect = self.params.calculate_defense_effectiveness(z)
        attack_effect = self.params.calculate_attack_effectiveness(w)

        # Calculate modified prevention probabilities
        P_mod = np.zeros((self.params.n, self.params.m))
        for i in range(self.params.n):
            for j in range(self.params.m):
                if x[j] == 1 and y[i] == 1:
                    effectiveness = self.params.P[i, j] * (1 + defense_effect[j]) / (1 + attack_effect[i])
                    P_mod[i, j] = min(max(effectiveness, 0), 1)  # Ensure value is in [0, 1]

        # Calculate maximum damage (without defenses)
        max_damage = np.sum(self.params.u * y)

        # Calculate prevented damage
        prevented_damage = 0
        for i in range(self.params.n):
            if y[i] == 1:
                # Find most effective defense for this attack
                best_prevention = 0
                for j in range(self.params.m):
                    if x[j] == 1:
                        best_prevention = max(best_prevention, P_mod[i, j])

                prevented_damage += self.params.u[i] * best_prevention

        # Calculate residual damage
        residual_damage = max_damage - prevented_damage

        # Fitness is the damage caused (attacker wants to maximize damage)
        fitness = residual_damage

        return fitness

    def calculate_defense_cost(self, defense: DefenseStrategy) -> float:
        """Calculate the total cost of a defense strategy."""
        cost = 0
        for j in range(self.params.m):
            if defense.x[j] == 1:
                cost += self.params.c_defense[j] + defense.z[j]
        return cost

    def calculate_attack_cost(self, attack: AttackStrategy) -> float:
        """Calculate the total cost of an attack strategy."""
        cost = 0
        for i in range(self.params.n):
            if attack.y[i] == 1:
                cost += self.params.c_attack[i] + attack.w[i]
        return cost

    def repair_defense_strategy(self, defense: DefenseStrategy) -> None:
        """Repair defense strategy to ensure it meets all constraints."""
        # Fix relationship between discrete and continuous variables
        for j in range(self.params.m):
            if defense.x[j] == 0:
                defense.z[j] = 0
            elif defense.z[j] > self.params.z_max[j]:
                defense.z[j] = self.params.z_max[j]

        # Fix budget constraint
        current_cost = self.calculate_defense_cost(defense)

        if current_cost > self.params.C_max_defense:
            # Try proportional reduction first
            if current_cost > 0:
                reduction_factor = self.params.C_max_defense / current_cost
                defense.z = defense.z * reduction_factor

            # Check if still over budget
            current_cost = self.calculate_defense_cost(defense)

            # If still over budget, remove least effective defenses
            if current_cost > self.params.C_max_defense:
                # Calculate effectiveness metric for each defense
                effectiveness = np.zeros(self.params.m)
                for j in range(self.params.m):
                    if defense.x[j] == 1:
                        # Sum of potential damage prevented
                        damage_prevented = np.sum(self.params.P[:, j] * self.params.u)
                        effectiveness[j] = damage_prevented / (self.params.c_defense[j] + defense.z[j])

                # Sort defenses by effectiveness (ascending)
                sorted_indices = np.argsort(effectiveness)

                # Remove defenses until under budget
                for j in sorted_indices:
                    if defense.x[j] == 1:
                        defense.x[j] = 0
                        defense.z[j] = 0
                        current_cost = self.calculate_defense_cost(defense)
                        if current_cost <= self.params.C_max_defense:
                            break

        # Optimize remaining budget
        if current_cost < self.params.C_max_defense:
            remaining_budget = self.params.C_max_defense - current_cost

            # Calculate marginal effectiveness
            marginal_effect = np.zeros(self.params.m)
            for j in range(self.params.m):
                if defense.x[j] == 1 and defense.z[j] < self.params.z_max[j]:
                    marginal_effect[j] = self.params.alpha_defense[j] * np.exp(
                        -self.params.alpha_defense[j] * defense.z[j])

            # Distribute remaining budget based on marginal effectiveness
            while remaining_budget > 0.1 and np.max(marginal_effect) > 0:
                # Select defense with highest marginal effectiveness
                j = np.argmax(marginal_effect)

                # Calculate maximum investment increment
                max_increment = min(remaining_budget, self.params.z_max[j] - defense.z[j])

                if max_increment <= 0:
                    marginal_effect[j] = 0
                    continue

                # Add investment
                defense.z[j] += max_increment
                remaining_budget -= max_increment

                # Update marginal effectiveness
                if defense.z[j] >= self.params.z_max[j]:
                    marginal_effect[j] = 0
                else:
                    marginal_effect[j] = self.params.alpha_defense[j] * np.exp(
                        -self.params.alpha_defense[j] * defense.z[j])

    def repair_attack_strategy(self, attack: AttackStrategy) -> None:
        """Repair attack strategy to ensure it meets all constraints."""
        # Fix relationship between discrete and continuous variables
        for i in range(self.params.n):
            if attack.y[i] == 0:
                attack.w[i] = 0
            elif attack.w[i] > self.params.w_max[i]:
                attack.w[i] = self.params.w_max[i]

        # Fix budget constraint
        current_cost = self.calculate_attack_cost(attack)

        if current_cost > self.params.C_max_attack:
            # Try proportional reduction first
            if current_cost > 0:
                reduction_factor = self.params.C_max_attack / current_cost
                attack.w = attack.w * reduction_factor

            # Check if still over budget
            current_cost = self.calculate_attack_cost(attack)

            # If still over budget, remove least effective attacks
            if current_cost > self.params.C_max_attack:
                # Calculate effectiveness metric for each attack
                effectiveness = np.zeros(self.params.n)
                for i in range(self.params.n):
                    if attack.y[i] == 1:
                        # Potential damage relative to cost
                        effectiveness[i] = self.params.u[i] / (self.params.c_attack[i] + attack.w[i])

                # Sort attacks by effectiveness (ascending)
                sorted_indices = np.argsort(effectiveness)

                # Remove attacks until under budget
                for i in sorted_indices:
                    if attack.y[i] == 1:
                        attack.y[i] = 0
                        attack.w[i] = 0
                        current_cost = self.calculate_attack_cost(attack)
                        if current_cost <= self.params.C_max_attack:
                            break

        # Optimize remaining budget
        if current_cost < self.params.C_max_attack:
            remaining_budget = self.params.C_max_attack - current_cost

            # Calculate marginal effectiveness
            marginal_effect = np.zeros(self.params.n)
            for i in range(self.params.n):
                if attack.y[i] == 1 and attack.w[i] < self.params.w_max[i]:
                    marginal_effect[i] = self.params.alpha_attack[i] * np.exp(
                        -self.params.alpha_attack[i] * attack.w[i]) * self.params.u[i]

            # Distribute remaining budget based on marginal effectiveness
            while remaining_budget > 0.1 and np.max(marginal_effect) > 0:
                # Select attack with highest marginal effectiveness
                i = np.argmax(marginal_effect)

                # Calculate maximum investment increment
                max_increment = min(remaining_budget, self.params.w_max[i] - attack.w[i])

                if max_increment <= 0:
                    marginal_effect[i] = 0
                    continue

                # Add investment
                attack.w[i] += max_increment
                remaining_budget -= max_increment

                # Update marginal effectiveness
                if attack.w[i] >= self.params.w_max[i]:
                    marginal_effect[i] = 0
                else:
                    marginal_effect[i] = self.params.alpha_attack[i] * np.exp(
                        -self.params.alpha_attack[i] * attack.w[i]) * self.params.u[i]

    def tournament_selection(self, population: List, tournament_size: int) -> Any:
        """Select individual using tournament selection."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover_defense(self, parent1: DefenseStrategy, parent2: DefenseStrategy) -> Tuple[
        DefenseStrategy, DefenseStrategy]:
        """Apply crossover to defense strategies."""
        if random.random() > self.crossover_prob:
            return DefenseStrategy(x=parent1.x.copy(), z=parent1.z.copy()), DefenseStrategy(x=parent2.x.copy(),
                                                                                            z=parent2.z.copy())

        # Two-point crossover for discrete variables
        n = len(parent1.x)
        point1, point2 = sorted(random.sample(range(n + 1), 2))

        child1_x = np.concatenate([parent1.x[:point1], parent2.x[point1:point2], parent1.x[point2:]])
        child2_x = np.concatenate([parent2.x[:point1], parent1.x[point1:point2], parent2.x[point2:]])

        # Arithmetic crossover for continuous variables
        alpha = random.random()
        child1_z = alpha * parent1.z + (1 - alpha) * parent2.z
        child2_z = (1 - alpha) * parent1.z + alpha * parent2.z

        # Create children and repair if needed
        child1 = DefenseStrategy(x=child1_x, z=child1_z)
        child2 = DefenseStrategy(x=child2_x, z=child2_z)

        self.repair_defense_strategy(child1)
        self.repair_defense_strategy(child2)

        return child1, child2

    def crossover_attack(self, parent1: AttackStrategy, parent2: AttackStrategy) -> Tuple[
        AttackStrategy, AttackStrategy]:
        """Apply crossover to attack strategies."""
        if random.random() > self.crossover_prob:
            return AttackStrategy(y=parent1.y.copy(), w=parent1.w.copy()), AttackStrategy(y=parent2.y.copy(),
                                                                                          w=parent2.w.copy())

        # Two-point crossover for discrete variables
        n = len(parent1.y)
        point1, point2 = sorted(random.sample(range(n + 1), 2))

        child1_y = np.concatenate([parent1.y[:point1], parent2.y[point1:point2], parent1.y[point2:]])
        child2_y = np.concatenate([parent2.y[:point1], parent1.y[point1:point2], parent2.y[point2:]])

        # Arithmetic crossover for continuous variables
        alpha = random.random()
        child1_w = alpha * parent1.w + (1 - alpha) * parent2.w
        child2_w = (1 - alpha) * parent1.w + alpha * parent2.w

        # Create children and repair if needed
        child1 = AttackStrategy(y=child1_y, w=child1_w)
        child2 = AttackStrategy(y=child2_y, w=child2_w)

        self.repair_attack_strategy(child1)
        self.repair_attack_strategy(child2)

        return child1, child2

    def mutate_defense(self, individual: DefenseStrategy) -> DefenseStrategy:
        """Apply mutation to defense strategy."""
        x_new = individual.x.copy()
        z_new = individual.z.copy()

        # Bit-flip mutation for discrete variables
        for j in range(self.params.m):
            if random.random() < self.mutation_prob:
                x_new[j] = 1 - x_new[j]

        # Gaussian mutation for continuous variables
        for j in range(self.params.m):
            if x_new[j] == 1 and random.random() < self.mutation_prob:
                # Standard deviation proportional to max investment
                sigma = self.params.z_max[j] * 0.1
                z_new[j] += np.random.normal(0, sigma)
                z_new[j] = max(0, min(z_new[j], self.params.z_max[j]))

        # Special mutation: complete strategy inversion (with low probability)
        if random.random() < 0.05:
            x_new = 1 - x_new

        # Create mutated individual and repair if needed
        mutated = DefenseStrategy(x=x_new, z=z_new)
        self.repair_defense_strategy(mutated)

        return mutated

    def mutate_attack(self, individual: AttackStrategy) -> AttackStrategy:
        """Apply mutation to attack strategy."""
        y_new = individual.y.copy()
        w_new = individual.w.copy()

        # Bit-flip mutation for discrete variables
        for i in range(self.params.n):
            if random.random() < self.mutation_prob:
                y_new[i] = 1 - y_new[i]

        # Gaussian mutation for continuous variables
        for i in range(self.params.n):
            if y_new[i] == 1 and random.random() < self.mutation_prob:
                # Standard deviation proportional to max investment
                sigma = self.params.w_max[i] * 0.1
                w_new[i] += np.random.normal(0, sigma)
                w_new[i] = max(0, min(w_new[i], self.params.w_max[i]))

        # Special mutation: complete strategy inversion (with low probability)
        if random.random() < 0.05:
            y_new = 1 - y_new

        # Create mutated individual and repair if needed
        mutated = AttackStrategy(y=y_new, w=w_new)
        self.repair_attack_strategy(mutated)

        return mutated

    def evolve(self) -> Tuple[DefenseStrategy, AttackStrategy]:
        """Run the coevolutionary process."""
        start_time = time.time()

        # Variables to track best strategies
        best_defense = None
        best_attack = None

        # Initial random champions
        defense_champion = random.choice(self.defense_pop)
        attack_champion = random.choice(self.attack_pop)

        # Main evolutionary loop
        for generation in range(self.max_gen):
            # Evaluate defense population against current attack champion
            for defense in self.defense_pop:
                defense.fitness = self.evaluate_defense_strategy(defense, attack_champion)

            # Update defense champion
            defense_champion = max(self.defense_pop, key=lambda ind: ind.fitness)

            # Evaluate attack population against new defense champion
            for attack in self.attack_pop:
                attack.fitness = self.evaluate_attack_strategy(attack, defense_champion)

            # Update attack champion
            attack_champion = max(self.attack_pop, key=lambda ind: ind.fitness)

            # Track best strategies
            if best_defense is None or defense_champion.fitness > best_defense.fitness:
                best_defense = DefenseStrategy(
                    x=defense_champion.x.copy(),
                    z=defense_champion.z.copy(),
                    fitness=defense_champion.fitness
                )

            if best_attack is None or attack_champion.fitness > best_attack.fitness:
                best_attack = AttackStrategy(
                    y=attack_champion.y.copy(),
                    w=attack_champion.w.copy(),
                    fitness=attack_champion.fitness
                )

            # Record history
            self.history['defense_best_fitness'].append(defense_champion.fitness)
            self.history['attack_best_fitness'].append(attack_champion.fitness)
            self.history['damage_values'].append(-defense_champion.fitness)  # Convert to damage
            self.history['defense_choices'].append(np.sum(defense_champion.x))
            self.history['attack_choices'].append(np.sum(attack_champion.y))

            # Print progress
            if generation % 10 == 0 or generation == self.max_gen - 1:
                elapsed_time = time.time() - start_time
                print(f"Generation {generation}, Time: {elapsed_time:.2f}s")
                print(
                    f"  Defense best fitness: {defense_champion.fitness:.2f}, Defenses used: {np.sum(defense_champion.x)}")
                print(
                    f"  Attack best fitness: {attack_champion.fitness:.2f}, Attacks used: {np.sum(attack_champion.y)}")
                print(f"  Damage value: {-defense_champion.fitness:.2f}")
                print("  Defense cost:", self.calculate_defense_cost(defense_champion))
                print("  Attack cost:", self.calculate_attack_cost(attack_champion))
                print()

            # Check for convergence
            if generation > 20:
                recent_values = self.history['damage_values'][-20:]
                if max(recent_values) - min(recent_values) < 0.01 * abs(recent_values[-1]):
                    print(f"Converged after {generation} generations.")
                    break

            # Create new populations
            new_defense_pop = []
            new_attack_pop = []

            # Elitism: keep best individuals
            elite_count = max(1, self.pop_size // 10)
            defense_elites = sorted(self.defense_pop, key=lambda ind: ind.fitness, reverse=True)[:elite_count]
            attack_elites = sorted(self.attack_pop, key=lambda ind: ind.fitness, reverse=True)[:elite_count]

            new_defense_pop.extend(defense_elites)
            new_attack_pop.extend(attack_elites)

            # Create the rest through selection, crossover, and mutation
            while len(new_defense_pop) < self.pop_size:
                parent1 = self.tournament_selection(self.defense_pop, self.tournament_size)
                parent2 = self.tournament_selection(self.defense_pop, self.tournament_size)

                child1, child2 = self.crossover_defense(parent1, parent2)

                child1 = self.mutate_defense(child1)
                child2 = self.mutate_defense(child2)

                new_defense_pop.append(child1)
                if len(new_defense_pop) < self.pop_size:
                    new_defense_pop.append(child2)

            while len(new_attack_pop) < self.pop_size:
                parent1 = self.tournament_selection(self.attack_pop, self.tournament_size)
                parent2 = self.tournament_selection(self.attack_pop, self.tournament_size)

                child1, child2 = self.crossover_attack(parent1, parent2)

                child1 = self.mutate_attack(child1)
                child2 = self.mutate_attack(child2)

                new_attack_pop.append(child1)
                if len(new_attack_pop) < self.pop_size:
                    new_attack_pop.append(child2)

            # Replace populations
            self.defense_pop = new_defense_pop
            self.attack_pop = new_attack_pop

            # Dynamically adjust mutation rate
            if generation % 10 == 0:
                # If converging, increase mutation to explore more
                if generation > 10:
                    recent_values = self.history['damage_values'][-10:]
                    if max(recent_values) - min(recent_values) < 0.05 * abs(recent_values[-1]):
                        self.mutation_prob = min(0.5, self.mutation_prob * 1.2)
                    else:
                        self.mutation_prob = max(0.05, self.mutation_prob * 0.9)

        # Final evaluation with best found strategies
        best_defense.fitness = self.evaluate_defense_strategy(best_defense, best_attack)
        best_attack.fitness = self.evaluate_attack_strategy(best_attack, best_defense)

        # Verify Nash equilibrium conditions (approximately)
        self.verify_nash_equilibrium(best_defense, best_attack)

        return best_defense, best_attack

    def verify_nash_equilibrium(self, defense: DefenseStrategy, attack: AttackStrategy) -> bool:
        """Verify if the given strategies form a Nash equilibrium."""
        print("\nVerifying Nash equilibrium conditions:")

        # Check defense strategy optimality
        is_defense_optimal = True
        defense_improvements = []

        # Check by flipping each defense mechanism
        for j in range(self.params.m):
            new_x = defense.x.copy()
            new_x[j] = 1 - new_x[j]

            new_defense = DefenseStrategy(x=new_x, z=defense.z.copy())
            self.repair_defense_strategy(new_defense)

            new_fitness = self.evaluate_defense_strategy(new_defense, attack)

            if new_fitness > defense.fitness:
                is_defense_optimal = False
                improvement = new_fitness - defense.fitness
                defense_improvements.append((j, improvement))

        # Check attack strategy optimality
        is_attack_optimal = True
        attack_improvements = []

        # Check by flipping each attack vector
        for i in range(self.params.n):
            new_y = attack.y.copy()
            new_y[i] = 1 - new_y[i]

            new_attack = AttackStrategy(y=new_y, w=attack.w.copy())
            self.repair_attack_strategy(new_attack)

            new_fitness = self.evaluate_attack_strategy(new_attack, defense)

            if new_fitness > attack.fitness:
                is_attack_optimal = False
                improvement = new_fitness - attack.fitness
                attack_improvements.append((i, improvement))

        # Report results
        if is_defense_optimal:
            print("Defense strategy is approximately optimal.")
        else:
            print("Defense strategy could be improved:")
            for j, improvement in sorted(defense_improvements, key=lambda x: -x[1])[:3]:
                print(f"  Changing defense {j} could improve fitness by {improvement:.4f}")

        if is_attack_optimal:
            print("Attack strategy is approximately optimal.")
        else:
            print("Attack strategy could be improved:")
            for i, improvement in sorted(attack_improvements, key=lambda x: -x[1])[:3]:
                print(f"  Changing attack {i} could improve fitness by {improvement:.4f}")

        is_nash = is_defense_optimal and is_attack_optimal
        if is_nash:
            print("The strategies approximately form a Nash equilibrium.")
        else:
            print("The strategies do not form a perfect Nash equilibrium, but may be close.")

        return is_nash

    def analyze_results(self, defense: DefenseStrategy, attack: AttackStrategy) -> Dict:
        """Analyze the results of the game."""
        results = {}

        # Basic results
        results['defense_strategy'] = defense
        results['attack_strategy'] = attack
        results['defense_fitness'] = defense.fitness
        results['attack_fitness'] = attack.fitness
        results['damage_value'] = -defense.fitness

        # Defense analysis
        defense_choices = np.where(defense.x == 1)[0]
        defense_investments = defense.z[defense_choices]
        defense_costs = self.params.c_defense[defense_choices]
        defense_total_costs = defense_costs + defense_investments

        results['defense_choices'] = defense_choices
        results['defense_investments'] = defense_investments
        results['defense_base_costs'] = defense_costs
        results['defense_total_costs'] = defense_total_costs
        results['defense_budget_usage'] = np.sum(defense_total_costs) / self.params.C_max_defense

        # If we have names for defense mechanisms, include them
        if hasattr(self.params, 'defense_names'):
            results['defense_names'] = [self.params.defense_names[j] for j in defense_choices]
        else:
            results['defense_names'] = [f"Defense {j}" for j in defense_choices]

        # Attack analysis
        attack_choices = np.where(attack.y == 1)[0]
        attack_investments = attack.w[attack_choices]
        attack_costs = self.params.c_attack[attack_choices]
        attack_damages = self.params.u[attack_choices]
        attack_total_costs = attack_costs + attack_investments

        results['attack_choices'] = attack_choices
        results['attack_investments'] = attack_investments
        results['attack_base_costs'] = attack_costs
        results['attack_total_costs'] = attack_total_costs
        results['attack_damages'] = attack_damages
        results['attack_budget_usage'] = np.sum(attack_total_costs) / self.params.C_max_attack

        # If we have names for attack vectors, include them
        if hasattr(self.params, 'attack_names'):
            results['attack_names'] = [self.params.attack_names[i] for i in attack_choices]
        else:
            results['attack_names'] = [f"Attack {i}" for i in attack_choices]

        # Effectiveness analysis
        defense_effect = self.params.calculate_defense_effectiveness(defense.z)
        attack_effect = self.params.calculate_attack_effectiveness(attack.w)

        # Calculate modified prevention matrix
        P_mod = np.zeros((self.params.n, self.params.m))
        for i in range(self.params.n):
            for j in range(self.params.m):
                if defense.x[j] == 1 and attack.y[i] == 1:
                    effectiveness = self.params.P[i, j] * (1 + defense_effect[j]) / (1 + attack_effect[i])
                    P_mod[i, j] = min(max(effectiveness, 0), 1)

        results['defense_effectiveness'] = defense_effect
        results['attack_effectiveness'] = attack_effect
        results['prevention_matrix'] = P_mod

        # Coverage analysis
        attack_coverage = np.zeros(self.params.n)
        for i in range(self.params.n):
            if attack.y[i] == 1:
                max_prevention = 0
                for j in range(self.params.m):
                    if defense.x[j] == 1:
                        max_prevention = max(max_prevention, P_mod[i, j])
                attack_coverage[i] = max_prevention

        results['attack_coverage'] = attack_coverage
        results['unprotected_attacks'] = np.where((attack.y == 1) & (attack_coverage < 0.2))[0]
        results['well_protected_attacks'] = np.where((attack.y == 1) & (attack_coverage > 0.8))[0]

        return results

    def visualize_results(self, results: Dict) -> None:
        """Visualize the results of the game."""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('DDoS Attack-Defense Game Analysis', fontsize=16)

        # Convergence plot
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.history['damage_values'], label='Damage')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Damage Value')
        ax1.set_title('Convergence of Damage Value')
        ax1.grid(True)

        # Strategy evolution
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.history['defense_choices'], label='Defenses Used')
        ax2.plot(self.history['attack_choices'], label='Attacks Used')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Count')
        ax2.set_title('Evolution of Strategy Complexity')
        ax2.legend()
        ax2.grid(True)

        # Defense investments
        defense_choices = results['defense_choices']
        defense_investments = results['defense_investments']
        defense_base_costs = results['defense_base_costs']
        defense_names = results['defense_names']

        ax3 = plt.subplot(2, 2, 3)
        bar_width = 0.35
        x = np.arange(len(defense_choices))
        ax3.bar(x, defense_base_costs, bar_width, label='Base Cost')
        ax3.bar(x, defense_investments, bar_width, bottom=defense_base_costs, label='Investment')
        ax3.set_xlabel('Defense Mechanism')
        ax3.set_ylabel('Cost')
        ax3.set_title('Defense Mechanism Costs and Investments')
        ax3.set_xticks(x)
        ax3.set_xticklabels(defense_names, rotation=45, ha='right')
        ax3.legend()

        # Attack coverage heatmap
        attack_choices = results['attack_choices']
        attack_names = results['attack_names']
        prevention_matrix = results['prevention_matrix']

        # Extract relevant part of the prevention matrix
        relevant_matrix = prevention_matrix[np.ix_(attack_choices, defense_choices)]

        ax4 = plt.subplot(2, 2, 4)
        im = ax4.imshow(relevant_matrix, cmap='viridis', aspect='auto')
        ax4.set_xlabel('Defense Mechanism')
        ax4.set_ylabel('Attack Vector')
        ax4.set_title('Attack Prevention Effectiveness')
        ax4.set_xticks(np.arange(len(defense_choices)))
        ax4.set_yticks(np.arange(len(attack_choices)))
        ax4.set_xticklabels(defense_names, rotation=45, ha='right')
        ax4.set_yticklabels(attack_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Prevention Probability')

        # Add text annotations to the heatmap
        for i in range(len(attack_choices)):
            for j in range(len(defense_choices)):
                text = ax4.text(j, i, f"{relevant_matrix[i, j]:.2f}",
                                ha="center", va="center", color="w" if relevant_matrix[i, j] < 0.5 else "black")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def ddos_defense_simulation(attack_vectors: int = 8, defense_mechanisms: int = 6) -> None:
    """Run a simulation of the DDoS attack-defense game."""
    print("=== DDoS Attack-Defense Optimization Using Discrete-Continuous Game ===")
    print(f"Attack vectors: {attack_vectors}, Defense mechanisms: {defense_mechanisms}")
    print("\nInitializing game parameters...")

    # Set random seed for reproducibility
    seed = 42

    # Initialize game parameters
    params = GameParameters(n_attacks=attack_vectors, m_defenses=defense_mechanisms, seed=seed)

    print("\nAttack types:")
    for i in range(params.n):
        print(f"  Attack {i}: Damage potential=${params.u[i]:.2f}, Base cost=${params.c_attack[i]:.2f}")

    print("\nDefense mechanisms:")
    for j in range(params.m):
        print(f"  Defense {j}: Base cost=${params.c_defense[j]:.2f}")

    print("\nInitializing solver...")
    solver = DiscreteContGameSolver(
        params=params,
        pop_size=50,
        max_gen=100,
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=3
    )

    print("\nStarting coevolution process...")
    best_defense, best_attack = solver.evolve()

    print("\n=== Final Results ===")
    print("Optimal Defense Strategy:")
    defense_indices = np.where(best_defense.x == 1)[0]
    for j in defense_indices:
        print(f"  Defense {j}: Base cost=${params.c_defense[j]:.2f}, Investment=${best_defense.z[j]:.2f}")

    print("\nOptimal Attack Strategy:")
    attack_indices = np.where(best_attack.y == 1)[0]
    for i in attack_indices:
        print(
            f"  Attack {i}: Damage potential=${params.u[i]:.2f}, Base cost=${params.c_attack[i]:.2f}, Investment=${best_attack.w[i]:.2f}")

    print(f"\nExpected damage: ${-best_defense.fitness:.2f}")
    print(
        f"Defense total cost: ${solver.calculate_defense_cost(best_defense):.2f} (Budget: ${params.C_max_defense:.2f})")
    print(f"Attack total cost: ${solver.calculate_attack_cost(best_attack):.2f} (Budget: ${params.C_max_attack:.2f})")

    # Analyze and visualize results
    results = solver.analyze_results(best_defense, best_attack)
    solver.visualize_results(results)

    print("\nSimulation completed.")


if __name__ == "__main__":
    ddos_defense_simulation(attack_vectors=8, defense_mechanisms=6)
