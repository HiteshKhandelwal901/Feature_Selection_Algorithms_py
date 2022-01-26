import random

dim = 7


class star:

    def __init__(self,name, fitness):
        self.name = name
        self.fitness = fitness
        self.pos = [self.random_generator_binary() for i in range(9)]

    def random_generator_binary(self):
        num = random.uniform(0, 1)
        if num > 0.5:
            return 1
        else:
            return 0

    def update_fitness(self):
        self.fitness = random.uniform(0,1)
        
def crossover(pop):
    
    new_pop = []
    name =0
    while len(new_pop)!= len(pop):
        print("len of new pop = ", len(new_pop))
        k = random.randrange(1,dim)
        rand_nums = random.sample(range(1, 4), 2)
        i = rand_nums[0]
        j = rand_nums[1]
        print("rand_nums = i= {} j = {} k = {}".format(i,j,k))
        print("star selected at random  = {} \n {}".format(pop[i].pos, pop[j].pos))
        
        new_star1 = star(str(name), 30)
        new_star2 = star(str(name+1), 45)
        new_star1.pos = pop[i].pos[:k] + pop[j].pos[k:]
        new_star2.pos = pop[j].pos[:k] + pop[i].pos[k:]
        print("new star1 pos {}".format(new_star1.pos))
        print("new star2 pos = {}".format(new_star2.pos))
        new_star1.update_fitness()
        new_star2.update_fitness()
        new_pop.append(new_star1)
        new_pop.append(new_star2)
        name = name+2

    stars_list = pop + new_pop
    sorted_stars_list = sorted(stars_list, key = lambda x: x.fitness, reverse= True)
    pop = sorted_stars_list[:4]
    print("length of sorted star list = ", len(pop))
    print("printing sorted fitness")
    for stars in sorted_stars_list:
        print(stars.fitness)
    print("length of stars_list", stars_list)


    return new_pop

if __name__ == "__main__":
    pop = []
    print("pop = ", pop)
    for i in range(4):
        print("i = ", i)
        pop.append(star(str(i), i+10))
        print("initalized star {} {}".format(pop[i].name, pop[i].pos))
    
    pop = crossover(pop)
    print("new pop = ", pop)
    for i in range(len(pop)):
        print(pop[i].pos, pop[i].name)


    