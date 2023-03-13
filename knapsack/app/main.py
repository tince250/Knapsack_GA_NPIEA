import numpy as np
import random
import time

# max knapsack capacity
CAPACITY = 0
# number of items
NUM_ITEMS = 0

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def __str__(self):
        return "(weight= " + str(self.weight) + ", value= " + str(self.value) + ")"

def read_file(filename):
    """Reads knapsack data from the file given."""

    global CAPACITY, NUM_ITEMS

    with open(filename) as f:
        # parsing header first
        header = f.readline()
        # "capacity: 6404180"
        CAPACITY = int(header.split(": ")[1])

        # list of loaded Item objects
        data = []
        while True:
            line = f.readline()
            if not line:
                break

            weight, value = line.strip().split(",")
            weight = int(weight)
            value = int(value)
            data.append(Item(weight, value))

        NUM_ITEMS = len(data)

    return data


def encode(unit, data):
    """Returns binary value representing items in backpack, 0 - not in it, 1 - in it."""

    bin_units = ""
    for i in data:
        if i in unit:
            bin_units += "1"
        else:
            bin_units += "0"
    return bin_units

def decode(unit, data):
    """Given the binary representation of the items in the backpack, returns the list of corresponding Item objects that are in it."""

    list = []
    i = 0
    for char in unit:
        if char == "1":
            list.append(data[i])
        i += 1

    return list


def cost_func(unit, data):
    """Returns the calculated sum of values and weights for the unit given (in binary rep)."""

    unit = decode(unit, data)
    total_value = 0
    total_weight = 0
    for item in unit:
        total_value += item.value
        total_weight += item.weight

    if total_weight <= CAPACITY:
        return total_weight, total_value
    else:
        return np.inf, -np.inf


def generate_initial_pop(data, size, condition, max_iter):
    """Returns and generates initial population as a list, with given size, of binary values of possibilities."""

    initial_pop = []
    iter = 0
    for i in range(size):
        while True:
            iter += 1
            value = ""
            for j in range(NUM_ITEMS):
                ran = random.random()
                if ran < condition:
                    value += "0"
                else:
                    value += "1"
            if value not in initial_pop and cost_func(value, data)[0] != np.inf:
                initial_pop.append(value)
                break



    return initial_pop

def roulette_selection(parents):
    """Returns list of sorted parent pairs chosen by roullete selection."""

    pairs = []
    i = 0
    for i in range(0, len(parents), 2):
        # list of calculated scores for selection of one pair
        scores = []

        # calculating scores
        for j in range(len(parents)):
            scores.append((len(parents)-j)*random.random())

        # selecting parents with max scores
        if (scores[0] >= scores[1]):
            max1_ind = 0
            max2_ind = 1
        else:
            max1_ind = 1
            max2_ind = 0

        for i in range(2, len(parents)):
            if scores[i] > scores[max1_ind]:
                max2_ind = max1_ind
                max1_ind = i
            elif scores[i] > scores[max2_ind]:
                max2_ind = 1

        pairs.append([parents[max1_ind], parents[max2_ind]])

    return pairs

def crossover(pairs):
    """Returns children of pairs of parents."""
    children = []

    for (i,j) in pairs:
        point = random.randrange(0, NUM_ITEMS)
        children.append(i[:point] + j[point:])
        children.append(j[:point] + i[point:])

    return children

def mutation(data, mutation_rate):
    """Single bit mutation on the entire popilation given."""

    # array of units after mutation
    mutated_data = []
    for unit in data:
        prob = random.random()
        if prob < mutation_rate:
            # units mutates
            p1 = random.randrange(0, NUM_ITEMS)
            # p2 = random.randrange(0, NUM_ITEMS)

            if unit[p1] == "1":
                mutated_unit = unit[:p1] + "0" + unit[p1+1:]
            else:
                mutated_unit = unit[:p1] + "1" + unit[p1 + 1:]
            # if p1 > p2:
            #     p1, p2 = p2, p1

            # mutated_unit = unit[:p1] + unit[p1:p2][::-1] + unit[p2:]

            mutated_data.append(mutated_unit)
        else:
            # unit doesn't mutate
            mutated_data.append(unit)

    return mutated_data

def sort_units(units, data):
    """Returns list of units sorted by total value."""

    return sorted(units, key = lambda x: cost_func(x, data)[1], reverse=True)

def elitism(parents, children, rate):
    """Returns a list of new generation with best choices."""

    per_num = int(np.round(len(parents) * rate))
    ret = parents[:per_num] + children[:(len(parents) - per_num)]
    return ret

def genetic_alg(data, mutation_rate = 0.8, elitis_rate=0.01, max_iter = 100, max_no_best_repeated = 50):
    """Driver function, complete GA algorithm."""

    # initialization

    parents = generate_initial_pop(data, 800, 0.5, 200)
    best_overall = ""
    best_repeated = 0
    for i in range(max_iter):
        parents = sort_units(parents, data)
        best_curr = parents[0]
        # selection
        pairs = roulette_selection(parents)
        # crossover
        children = crossover(pairs)
        # mutation
        children = mutation(children, mutation_rate)
        # elitism
        children = sort_units(children, data)
        parents = elitism(parents, children, elitis_rate)

        if cost_func(best_curr, data)[1] > cost_func(best_overall, data)[1]:
            best_overall = best_curr
        elif cost_func(best_curr, data)[1] == cost_func(best_overall, data)[1]:
            best_repeated += 1

        best_weight, best_val = cost_func(parents[0], data)

        if best_repeated > max_no_best_repeated:
            print("Due to the population convergence, the algorithm stopped.")
            best_decoded = decode(parents[0], data)
            best_weight, best_val = cost_func(parents[0], data)
            return best_val, best_weight, best_decoded

    #print("Max iteration count reached.")
    best_decoded = decode(parents[0], data)
    best_weight, best_val = cost_func(parents[0], data)
    return best_val, best_weight, best_decoded

if __name__ == "__main__":

    data = read_file("../data/data_knapsack01.txt")
    start = time.time()
    best_val, weight, items = genetic_alg(data)
    end = time.time()
    print('Evaluation time: {}s'.format(round(end - start, 7)))

    print("Best value: ",  best_val)
    print("Weight: ", weight)
    print("\nList of picked items: \n")
    it = 0
    for item in items:
        it += 1
        print("Item number: ", it,", value of ", str(item.value) + "$, and weight of ", str(item.weight) + "kg.")

    # data = read_file("../data/data_knapsack01.txt")
    # for item in data:
    #     print(item)
    # print(CAPACITY)
    # print(NUM_ITEMS)
    #
    # list = [data[5], data[3], data[0]]
    # encoded = encode(list, data)
    # print(cost_func(encoded, data))
    # for item in decode(encoded, data):
    #     print(item)
    # parents = generate_initial_pop(data, 20, 0.5, 200)
    # pairs = roulette_selection(parents)

