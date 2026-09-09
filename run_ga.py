from genetic_algorithm import GeneticAlgorithm
from ga_data import data


# ==========================================
# MEMBUAT GENETIC ALGORITHM
# ==========================================

ga = GeneticAlgorithm(
    population_size=20,
    generations=30,
    mutation_rate=0.1
)


# ==========================================
# MENJALANKAN GA
# ==========================================

best_solution, best_fitness = ga.run(data)


# ==========================================
# HASIL
# ==========================================

print("\n")
print("======================================")
print(" HASIL OPTIMASI GENETIC ALGORITHM")
print("======================================")

print(
    f"Fire Threshold  : {best_solution[0]:.3f}"
)

print(
    f"Smoke Threshold : {best_solution[1]:.3f}"
)

print(
    f"Fitness         : {best_fitness:.3f}"
)

print("======================================")