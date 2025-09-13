import numpy as np
from ga import GeneticAlgorithm
from pso import ParticleSwarmOptimization
from benchmark_functions import get_benchmarks, split_unimodal_multimodal

RUNS = 20
POP = 50
BUDGET = 40000

def evaluate(bench, seed_base=0):
    pso_vals, ga_vals = [], []
    for r in range(RUNS):
        pso = ParticleSwarmOptimization(bench.func, bench.dim, bench.bounds, pop_size=POP, budget=BUDGET, seed=1000+seed_base+r)
        ga  = GeneticAlgorithm(bench.func, bench.dim, bench.bounds, pop_size=POP, budget=BUDGET, seed=2000+seed_base+r)
        pso_vals.append(pso.run())
        ga_vals.append(ga.run())
    pso_vals, ga_vals = np.array(pso_vals), np.array(ga_vals)
    return (pso_vals.mean(), pso_vals.std()), (ga_vals.mean(), ga_vals.std())


def md_info_table(title, benches):
    lines = [
        f"- {title} -",
        "| No. | Function name | PSO (Mean ± Std) | GA (Mean ± Std) |",
        "",
        "------------------------------------------------------------",
    ]
    for i, b in enumerate(benches, start=1):
        lb = np.min(b.bounds[0])
        ub = np.max(b.bounds[1])
        lines.append(f"| {i} | {b.name} | {b.dim} | [{lb:g}, {ub:g}] | {b.category} |")
    return "\n".join(lines)

def md_result_table(title, benches, stats):
    lines = [
        f"- {title} -",
        "",
        "| No. | Function name | PSO (Mean ± Std) | GA (Mean ± Std) |",
        "------------------------------------------------------------",
    ]
    for i, (b, st) in enumerate(zip(benches, stats), start=1):
        (pm, ps), (gm, gs) = st
        lines.append(f"| {i} | {b.name} | {pm:.6g} ± {ps:.3g} | {gm:.6g} ± {gs:.3g} |")
    return "\n".join(lines)

def main():
    B = get_benchmarks()
    uni, multi = split_unimodal_multimodal(B)

    uni_stats = [evaluate(b, seed_base=0) for b in uni]
    multi_stats = [evaluate(b, seed_base=5000) for b in multi]

    print(md_info_table("unimodal", uni))
    print()
    print(md_info_table("multimodal", multi))
    print()
    print(md_result_table("result unimodal ", uni, uni_stats))
    print()
    print(md_result_table("result multimodal", multi, multi_stats))


if __name__ == "__main__": 
    main()

# def main(target_name="Ackley"):
#     B = get_benchmarks()

#     if target_name not in B:
#         print(f"Benchmark '{target_name}' not found.")
#         return

#     b = B[target_name]
#     stats = evaluate(b, seed_base=0 if b.category == "unimodal" else 5000)

#     print(md_info_table(f"{b.category} - {b.name}", [b]))
#     print()
#     print(md_result_table(f"Result - {b.name}", [b], [stats]))

# if __name__ == "__main__":
#     main("Ackley")  


