import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Union
from math import pi

@dataclass
class Benchmark:
    name: str
    func: Callable[[np.ndarray], float]
    dim: int
    bounds: Tuple[np.ndarray, np.ndarray]
    category: str  # 'unimodal' or 'multimodal'

def _mk_bounds(dim: int, low: Union[float, List[float]], high: Union[float, List[float]]):
    lb = np.array(low if isinstance(low, list) else [low]*dim, dtype=float)
    ub = np.array(high if isinstance(high, list) else [high]*dim, dtype=float)
    return lb, ub

def sphere(x): return float(np.sum(x**2))
def rosenbrock(x):
    xi = x[:-1]; xnext = x[1:]
    return float(np.sum(100*(xnext - xi**2)**2 + (1 - xi)**2))
def rastrigin(x): return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))
def ackley(x):
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    return float(-a*np.exp(-b*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.e)
def griewank(x):
    i = np.arange(1, len(x)+1, dtype=float)
    return float(np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(i))) + 1)
def schwefel_223(x): return float(np.sum(x**10))
def schwefel_220(x): return float(np.sum(np.abs(x)))
def schwefel_221(x): return float(np.max(np.abs(x)))
def schwefel_222(x):
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))
def schwefel(x):
    return float(418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x)))))
def zakharov(x):
    i = np.arange(1, len(x)+1, dtype=float)
    s1 = np.sum(x**2)
    s2 = np.sum(0.5*i*x)
    return float(s1 + s2**2 + s2**4)
def sum_squares(x):
    i = np.arange(1, len(x)+1, dtype=float)
    return float(np.sum(i*(x**2)))
def quartic_noise(x):
    rng = np.random.default_rng(12345)  # fixed noise seed for reproducibility
    return float(np.sum(np.arange(1, len(x)+1) * (x**4)) + rng.random())
def salomon(x):
    r = np.sqrt(np.sum(x**2))
    return float(1 - np.cos(2*np.pi*r) + 0.1*r)
def ridge(x):  
    cumsum = np.cumsum(x)
    return float(np.sum(cumsum**2))
def trid(x):
    n = len(x)
    i = np.arange(1, n+1, dtype=float)
    return float(np.sum((x-1)**2) - np.sum(x[:-1]*x[1:]))
def xin_she_yang(x):
    return float(np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2))))
def xin_she_yang_n2(x):
    return float(np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2))) + np.random.random()*0.1)
def xin_she_yang_n3(x, a=5, b=15):
    return float(np.exp(-np.sum((x/ a)**2)) - 2*np.exp(-np.sum(((x - b)/ a)**2)))
def xin_she_yang_n4(x):
    return float(np.sum(np.sin(x)**2 - np.exp(-x**2)) * np.exp(-np.sum(np.sin(np.arange(1, len(x)+1)*x**2)/np.pi)))
def styblinski_tank(x): return float(np.sum(x**4 - 16*x**2 + 5*x)/2)
def brown(x):
    xi = x[:-1]; xnext = x[1:]
    return float(np.sum((xi**2)**(xnext**2 + 1) + (xnext**2)**(xi**2 + 1)))
def leon(x): return float(100*(x[1]-x[0]**2)**2 + (1-x[0])**2)
def matyas(x): return float(0.26*np.sum(x**2) - 0.48*x[0]*x[1])
def beale(x):
    x1, x2 = x[0], x[1]
    return float((1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2)
def brent(x):
    x1, x2 = x[0], x[1]
    return float((x1+10)**2 + (x2+10)**2 + np.exp(-x1**2 - x2**2))
def bohachevsky1(x):
    x1, x2 = x[0], x[1]
    return float(x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7)
def bohachevsky2(x):
    x1, x2 = x[0], x[1]
    return float(x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1)*np.cos(4*np.pi*x2) + 0.3)
def booth(x):
    x1, x2 = x[0], x[1]
    return float((x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2)
def three_hump_camel(x):
    a, b = x[0], x[1]
    return float(2*a**2 - 1.05*a**4 + a**6/6 + a*b + b**2)
def egg_crate(x):
    x1, x2 = x[0], x[1]
    return float(x1**2 + x2**2 + 25*(np.sin(x1)**2 + np.sin(x2)**2))
def cross_in_tray(x):
    x1, x2 = x[0], x[1]
    fact = np.abs(100 - np.sqrt(x1**2 + x2**2)/np.pi)
    return float(-0.0001*(np.abs(np.sin(x1)*np.sin(x2)*np.exp(fact)) + 1)**0.1)
def holder_table(x):
    x1, x2 = x[0], x[1]
    return float(-np.abs(np.sin(x1)*np.cos(x2)*np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2)/np.pi))))
def easom(x):
    x1, x2 = x[0], x[1]
    return float(-np.cos(x1)*np.cos(x2)*np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2)))
def adjiman(x):
    x1, x2 = x[0], x[1]
    return float(np.cos(x1)*np.sin(x2) - x1/(x2**2 + 1))
def ackley_n2(x):
    x1, x2 = x[0], x[1]
    return float(-200*np.exp(-0.02*np.sqrt(x1**2 + x2**2)) + 5*np.exp(np.cos(3*x1)+np.sin(3*x2)))
def ackley_n3(x):
    x1, x2 = x[0], x[1]
    return float(0.5 + (np.sin(x1)**2 + np.sin(x2)**2 - 0.5) / (1 + 0.001*(x1**2 + x2**2))**2)
def ackley_n4(x):
    # generalization to n-D per reference family
    return ackley(x)
def periodic(x):
    return float(1 + np.sum(np.sin(x)**2) - 0.1*np.exp(-np.sum(x**2)))
def qing(x):
    i = np.arange(1, len(x)+1, dtype=float)
    return float(np.sum((x**2 - i)**2))
def salomon2(x):  
    return salomon(x)
def schaffer_n2(x):
    x1, x2 = x[0], x[1]
    num = (np.sin(x1**2 - x2**2))**2 - 0.5
    den = (1 + 0.001*(x1**2 + x2**2))**2
    return float(0.5 + num/den)
def schaffer_n4(x):
    x1, x2 = x[0], x[1]
    num = (np.cos(np.sin(np.abs(x1**2 - x2**2)))**2) - 0.5
    den = (1 + 0.001*(x1**2 + x2**2))**2
    return float(0.5 + num/den)
def schaffer_n1(x):
    x1, x2 = x[0], x[1]
    return float(0.5 + (np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5) / (1 + 0.001*(x1**2 + x2**2))**2)
def schaffer_n3(x):
    x1, x2 = x[0], x[1]
    return float(0.5 + (np.cos(np.sin(np.abs(x1**2 - x2**2))) - 0.5) / (1 + 0.001*(x1**2 + x2**2))**2)
def drop_wave(x):
    x = np.asarray(x)
    sum_sq = np.sum(x**2)
    numerator = 1 + np.cos(12 * np.sqrt(sum_sq))
    denominator = 0.5 * sum_sq + 2
    return -numerator / denominator
def alpine_n1(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))
def alpine_n2(x):
    x = np.asarray(x)
    return float(np.prod(np.sqrt(x) * np.sin(x)))
def bird(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float(np.sin(x[0]) * np.exp((1 - np.cos(x[1]))**2) +np.cos(x[1]) * np.exp((1 - np.sin(x[0]))**2) + (x[0] - x[1])**2)
def gramacy_lee(x):
    x = np.asarray(x)
    assert len(x) == 1
    return float(np.sin(10 * np.pi * x[0]) / (2 * x[0]) + (x[0] - 1)**4)
def deckkers_aarts(x):
    x = np.asarray(x)
    assert len(x) == 2
    r2 = x[0]**2 + x[1]**2
    return float(1e5 * x[0]**2 + x[1]**2 - r2**2 + 1e-5 * r2**4)
def levi(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float(np.sin(3 * np.pi * x[0])**2 + (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2) + (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2))
def wolfe(x):
    x = np.asarray(x)
    assert len(x) == 3
    return float((4/3) * ((x[0]**2 + x[1]**2 - x[0]*x[1])**0.75) + x[2])
def el_attar(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float((x[0]**2 + x[1] - 10)**2 + (x[0] + x[1]**2 - 7)**2 + (x[0]**2 + x[1]**3 - 1)**2)
def forrester(x):
    x = np.asarray(x)
    assert len(x) == 1
    return float((6 * x[0] - 2)**2 * np.sin(12 * x[0] - 4))
def keaneN(x):
    x = np.asarray(x)
    return float(np.prod(np.sqrt(x) * np.sin(x)))
def shubert(x):
    x = np.asarray(x)
    result = 1
    for i in range(len(x)):
        inner = sum(j * np.cos((j + 1) * x[i] + j) for j in range(1, 6))
        result *= inner
    return float(result)
def shubert_n4(x):
    x = np.asarray(x)
    result = 0
    for i in range(len(x)):
        result += sum(np.cos((j + 1) * x[i] + j) for j in range(1, 6))
    return float(result)
def shubert_3(x):
    x = np.asarray(x)
    result = 0
    for i in range(len(x)):
        for j in range(1, 6):
            result += j * np.sin((j + 1) * x[i] + j)
    return float(result)
def mccormick(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float(np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1)
def goldstein_price(x):
    x = np.asarray(x)
    assert len(x) == 2
    part1 = (1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2))
    part2 = (30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
    return float(part1 * part2)
def himmelblau(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)
def happy_cat(x):
    x = np.asarray(x)
    n = len(x)
    norm_sq = np.linalg.norm(x)**2
    term1 = ((norm_sq - n)**2)**0.25
    term2 = (0.5 * norm_sq + np.sum(x)) / n
    return float(term1 + term2 + 0.5)
def carrom_table(x):
    x = np.asarray(x)
    assert len(x) == 2
    r = np.sqrt(x[0]**2 + x[1]**2)
    return float(-1/30 * np.exp(2 * abs(1 - r / np.pi)) * (np.cos(x[0])**2) * (np.cos(x[1])**2))
def bukin_n6(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float(100 * np.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10))
def bartels_conn(x):
    x = np.asarray(x)
    assert len(x) == 2
    return float(abs(x[0]**2 + x[1]**2 + x[0]*x[1]) + abs(np.sin(x[0])) + abs(np.cos(x[1])))
def powell_sum(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x)**(np.arange(1, len(x)+1) + 1)))
def schwefel_222(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))
def schwefel_221(x):
    x = np.asarray(x)
    return float(np.max(np.abs(x)))
def schwefel_220(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x)))

def get_benchmarks() -> Dict[str, Benchmark]:
    B = {}

    def add(name, func, dim, low, high, category):
        lb, ub = _mk_bounds(dim, low, high)
        B[name] = Benchmark(name, func, dim, (lb, ub), category)

    add("Sphere", sphere, 30, -100, 100, "unimodal")
    add('Drop-wave', drop_wave, 2, -5, 5, 'unimodal')
    add("Ridge", ridge, 30, -100, 100, "unimodal")
    add("Sum Squares", sum_squares, 30, -10, 10, "unimodal")
    add("Schwefel 2.23", schwefel_223, 30, -100, 100, "unimodal")
    add("Schwefel 2.22", schwefel_222, 30, -100, 100, "unimodal")
    add("Schwefel 2.21", schwefel_221, 30, -100, 100, "unimodal")
    add("Schwefel 2.20", schwefel_220, 30, -100, 100, "unimodal")
    add("Zakharov", zakharov, 30, -5, 10, "unimodal")
    add("Brown", brown, 30, -4, 4, "unimodal")
    add("Trid", trid, 30, -100, 100, "unimodal")
    add("Matyas", matyas, 2, -10, 10, "unimodal")
    add("Brent", brent, 2, -10, 10, "unimodal")
    add("Schaffer N.1", schaffer_n1, 2, -100, 100, "unimodal")
    add("Schaffer N.2", schaffer_n2, 2, -100, 100, "unimodal")
    add("Schaffer N.3", schaffer_n3, 2, -100, 100, "unimodal")
    add("Schaffer N.4", schaffer_n4, 2, -100, 100, "unimodal")
    add("Ackley N.2", ackley_n2, 2, -1, 1, "unimodal")
    add("Powell Sum", powell_sum, 30, -1, 1, "unimodal")
    add("Booth", booth, 2, -10, 10, "unimodal")
    add("Griewank", griewank, 30, -600, 600, "unimodal")  
    add('leon', leon, 2, -1.5, 1.5 ,"unimodal")  
    add("Xin-She Yang N.3", xin_she_yang_n3, 30, -5, 5, "unimodal")
    add("Exponential", lambda x: float(np.exp(np.sum(x)/len(x)) - 1), 30, -1, 1, "unimodal")
    add("Three-Hump Camel", three_hump_camel, 2, -2, 2, "unimodal")
    add("Bohachevsky N.1", bohachevsky1, 2, -100, 100, "unimodal")
    add("Rastrigin", rastrigin, 30, -5.12, 5.12, "multimodal")
    add("Ackley", ackley, 30, -32.768, 32.768, "multimodal")
    add("Ackley N.3", ackley_n3, 2, -1, 1, "multimodal")
    add("Ackley N.4", ackley_n4, 30, -1, 1, "multimodal")
    add("Alpine N.1", alpine_n1, 30, -10, 10, "multimodal")
    add("Alpine N.2", alpine_n2, 30, 0, 10, "multimodal")
    add("Schwefel", schwefel, 30, -500, 500, "multimodal")
    add("Quartic Noise", quartic_noise, 30, -1.28, 1.28, "multimodal")
    add("Salomon", salomon, 30, -100, 100, "multimodal")
    add("Qing", qing, 30, -500, 500, "multimodal")
    add("Periodic", periodic, 30, -5.12, 5.12, "multimodal")
    add("Bird", bird, 2, -2*np.pi, 2*np.pi, "multimodal")
    add("Deckkers-Aarts", deckkers_aarts, 2, -20, 20, "multimodal")
    add("Gramcy Lee", gramacy_lee, 1, -0.5, 2.5, "multimodal")
    add("Levi", levi, 2, -10, 10, "multimodal")
    add("El Attar", el_attar, 2, -500, 500, 'multimodal')
    add("Wolfe", wolfe, 3, 0, 2, "multimodal")
    add("Keane", keaneN, 2, 0, 10, "multimodal")
    add("Burtels conn", bartels_conn, 2, -500, 500, "multimodal")
    add("Carrom Table", carrom_table, 2, -10, 10, "multimodal")
    add("Bukin N.6",bukin_n6, 2, [-15, -5], [-3, 3], "multimodal")
    add("Forrester", forrester, 1, -0.5, 2.5, 'multimodal')
    add("shubert", shubert, 30, -10, 10, "multimodal")
    add("shubert-3", shubert_3, 30, -10, 10, "multimodal")
    add("shubert N.4", shubert_n4, 30, -10, 10, "multimodal")
    add("Styblinski-Tank", styblinski_tank, 30, -5, 5, "multimodal")
    add("Xin-She Yang", xin_she_yang, 30, -5, 5, "multimodal")
    add("Xin-She Yang N.2", xin_she_yang_n2, 30, -2, 2, "multimodal")
    add("Xin-She Yang N.4", xin_she_yang_n4, 30, -1, 1, "multimodal")
    add("Rosenbrock", rosenbrock, 30, -2.048, 2.048, "multimodal")
    add("Adjiman", adjiman, 2, [-1, -1], [2, 1], "multimodal")
    add("Bohachevsky N.2", bohachevsky2, 2, -100, 100, "multimodal")
    add("Egg Crate", egg_crate, 2, -2*np.pi, 2*np.pi, "multimodal")
    add("Cross-in-Tray", cross_in_tray, 2, -10, 10, "multimodal")
    add("Holder-Table", holder_table, 2, -10, 10, "multimodal")
    add("Easom", easom, 2, -100, 100, "multimodal")
    add("Beale", beale, 2, -4.5, 4.5, "multimodal")
    add("McCormick", mccormick, 2, [-1.5, -3], [4, 4], "multimodal")
    add("Goldstein-Price", goldstein_price, 2, -2, 2, "multimodal")
    add("Himmelblau",himmelblau, 2, -6, 6, "multimodal")
    add("Happy Cat",happy_cat, 30, -2, 2, "multimodal")


    return B

def split_unimodal_multimodal(B: Dict[str, Benchmark]):
    uni = [b for b in B.values() if b.category == "unimodal"]
    multi = [b for b in B.values() if b.category == "multimodal"]
    return uni, multi
