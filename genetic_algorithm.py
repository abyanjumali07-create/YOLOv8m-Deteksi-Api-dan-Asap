import random


class GeneticAlgorithm:

    def __init__(
        self,
        population_size=20,
        generations=30,
        mutation_rate=0.1
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    # ==========================================
    # 1. MEMBUAT POPULASI AWAL
    # ==========================================

    def create_population(self):

        population = []

        for _ in range(self.population_size):

            fire_threshold = random.uniform(0.3, 0.9)
            smoke_threshold = random.uniform(0.3, 0.9)

            chromosome = [
                fire_threshold,
                smoke_threshold
            ]

            population.append(chromosome)

        return population

    # ==========================================
    # 2. FITNESS FUNCTION
    # ==========================================

    def fitness(self, chromosome, data):

        fire_threshold = chromosome[0]
        smoke_threshold = chromosome[1]

        TP = 0
        FP = 0
        FN = 0

        for item in data:

            confidence = item["confidence"]
            actual = item["actual"]

            if item["class"] == "fire":
                threshold = fire_threshold
            else:
                threshold = smoke_threshold

            prediction = confidence >= threshold

            if prediction and actual:
                TP += 1

            elif prediction and not actual:
                FP += 1

            elif not prediction and actual:
                FN += 1

        # Jika tidak ada True Positive
        if TP == 0:
            return 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        if precision + recall == 0:
            return 0

        f1 = 2 * precision * recall / (precision + recall)

        return f1

    # ==========================================
    # 3. SELECTION
    # ==========================================

    def selection(self, population, data):

        population.sort(
            key=lambda x: self.fitness(x, data),
            reverse=True
        )

        # Ambil 50% terbaik
        survivors = population[:self.population_size // 2]

        return survivors

    # ==========================================
    # 4. CROSSOVER
    # ==========================================

    def crossover(self, parent1, parent2):

        child = [

            (parent1[0] + parent2[0]) / 2,

            (parent1[1] + parent2[1]) / 2

        ]

        return child

    # ==========================================
    # 5. MUTATION
    # ==========================================

    def mutation(self, chromosome):

        chromosome = chromosome.copy()

        # Mutasi Fire
        if random.random() < self.mutation_rate:

            chromosome[0] += random.uniform(-0.05, 0.05)

        # Mutasi Smoke
        if random.random() < self.mutation_rate:

            chromosome[1] += random.uniform(-0.05, 0.05)

        # Batasi nilai threshold
        chromosome[0] = max(
            0.3,
            min(0.9, chromosome[0])
        )

        chromosome[1] = max(
            0.3,
            min(0.9, chromosome[1])
        )

        return chromosome

    # ==========================================
    # 6. PROSES GENETIC ALGORITHM
    # ==========================================

    def run(self, data):

        # Buat populasi awal
        population = self.create_population()

        best_solution = None
        best_fitness = 0

        # Loop generasi
        for generation in range(self.generations):

            # Selection
            population = self.selection(
                population,
                data
            )

            # Solusi terbaik generasi sekarang
            current_best = population[0]

            current_fitness = self.fitness(
                current_best,
                data
            )

            # Simpan solusi terbaik
            if current_fitness > best_fitness:

                best_fitness = current_fitness

                best_solution = current_best.copy()

            # Tampilkan proses
            print(
                f"Generasi {generation + 1:02d} | "
                f"Fire = {current_best[0]:.3f} | "
                f"Smoke = {current_best[1]:.3f} | "
                f"Fitness = {current_fitness:.3f}"
            )

            # Buat populasi baru
            new_population = population.copy()

            while len(new_population) < self.population_size:

                parent1 = random.choice(population)

                parent2 = random.choice(population)

                # Crossover
                child = self.crossover(
                    parent1,
                    parent2
                )

                # Mutation
                child = self.mutation(child)

                new_population.append(child)

            population = new_population

        return best_solution, best_fitness