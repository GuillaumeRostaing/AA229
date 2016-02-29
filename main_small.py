#import
import frs
from random import randint
import hsolv

#world
gamma = 0.95
grid_size = 6 #initial grid size
nb_rocks = 6 #nb rocks
rock_positions = []
rock_goodness = []
for i in range(nb_rocks):
    b = [randint(0,grid_size-1),randint(0,grid_size-1)]
    while b in rock_positions:
        b = [randint(0,grid_size-1),randint(0,grid_size-1)]
    rock_positions.append(b)
    rock_goodness.append(randint(0,1))

if all(item == 0 for item in rock_goodness):
    rock_goodness[randint(0,nb_rocks-1)] = 1
rock_goodness = [1, 0, 0, 1, 1, 0]
rock_positions = [[0, 4], [2, 2], [2, 4], [3, 1], [3, 4], [4, 1]]

#multi agents
agents = [2,1] #agents' speed
init_agents_position = [[0,0],[0,2]]
init_agents_action = [["U","D","L","R"],["U","D","L","R","S"]]

#hsolv
main_sim = hsolv.HSolv()
main_sim.Generate(grid_size,agents,init_agents_position,init_agents_action,rock_positions,rock_goodness,gamma)
main_sim.Solve()
print main_sim.cumulative_reward
print main_sim.iter
