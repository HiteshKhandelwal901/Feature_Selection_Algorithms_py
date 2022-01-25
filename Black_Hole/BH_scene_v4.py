import random

def crossover(pop):
    
    new_pop = []
    while len(new_pop)!= len(pop):
        print("len of new pop = ", len(new_pop))
    star1 = pop[0]
    star2 = pop[1]
    print("star1 pos = ", star1.pos)
    print("star2 pos = ", star2.pos)
    I = random.randrange(1,7)
    print("J = ", I)
    j = 7
    star1_first_half = pop[0].pos[:j]
    star1_second_half = pop[0].pos[j:]
    star2_first_half = pop[1].pos[:j]
    star2_second_half = pop[1].pos[j:]
    print("first half seocnd half of star1  = {} {}".format(star1_first_half,star1_second_half ))
    print("first half seocnd half of star2  = {} {}".format(star2_first_half,star2_second_half ))
    
    new_star1 = star("new_star1", 30)
    new_star2 = star("new_star2", 45)
    new_star1.pos = star1_first_half + star2_second_half
    new_star2.pos = star1_first_half + star2_second_half
    print("new star1 pos {}".format(new_star1.pos))
    print("new star2 pos = {}".format(new_star2.pos))
    new_pop.append(star1)
    new_pop.append(star2)

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
        

if __name__ == "__main__":
    pop = []
    for i in range(4):
        pop[i].append(star(i, i+10))
        print("initalized star {} {}".format(pop[i].name, pop[i].pos))
    
    pop = crossover(pop)
    print("new pop = ", pop)
    for i in range(len(pop)):
        print(pop[i].pos, pop[i].name)


    