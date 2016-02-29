#import
import rollouts
import frs
from random import randint
import time

#initiate problem
gamma = 0.95

#simulation parameters
nb_iter = 200 #nb iterations
depth = 7 #depth
nb_simulations = 10
max_reward = -10000
min_reward = 0
average_reward = 0
average_distance = 0
fail = 0
success = 0
best_policy = []
best_distance = 0
start_time = time.time()
for nb_simu in range(nb_simulations):
    #multi agents
    agents = [1]
    #agents = [2,1]
    init_agents_position = [[0,0]]
    #init_agents_position = [[0,0],[0,2]]
    init_agents_action = [["U","D","L","R","S"]]
    #init_agents_action = [["U","D","L","R"],["U","D","L","R","S"]]

    #parameters
    grid_size = 6 #initial grid size
    nb_rocks = 6
    #rock_position = [[1,1]]
    #rock_position = [[0,0],[1,1]]
    #rock_position = [[1,0],[2,2],[1,1]]
    #rock_position = [[3,1],[2,1],[3,3],[0,1]]
    #rock_goodness = [1]
    #rock_goodness = [1,0,1]
    rock_goodness = [1, 0, 0, 1, 1, 0]
    rock_position = [[0, 4], [2, 2], [2, 4], [3, 1], [3, 4], [4, 1]]

    #set up framework
    FRS = frs.FRS()
    FRS.Generate(grid_size,agents,init_agents_position,init_agents_action,rock_position,rock_goodness)
    agents_action = []
    for agent in FRS.agents:
        agents_action.append(agent.list_of_actions)

    #Monte Carlo Tree Search (factored statistics)
    belief_goodness = list(FRS.belief_state)
    belief_position = list(FRS.belief_position)
    total_reward = 0
    end = 0
    i = 0
    policy = []
    while (end == 0) and (i < 40):
        i += 1

        #position of the robots
        pos = []
        for agent in FRS.agents:
            pos.append(agent.current_position)

        #copy belief
        blf_goodness = list(belief_goodness)
        blf_position = list(belief_position)

        #determine next joint action
        init_agents_position = []
        for agent in FRS.agents:
            init_agents_position.append(agent.current_position)
        RO = rollouts.Rollouts()
        RO.InitiateParameters(grid_size,agents,init_agents_position,init_agents_action,agents_action,len(rock_position),gamma)
        next_action = RO.SelectAction(blf_goodness,blf_position,pos,depth,nb_iter,True,"legal")
        policy.append(next_action)

        #take next action
        for agent_id in range(FRS.nb_agents):
            action = next_action[agent_id]
            if isinstance(action,int) == True:
                action = [action]
            result = FRS.TakeAction(action,agent_id)
        total_reward += (gamma**i)*result[1]

        #update and retrieve belief
        FRS.BeliefUpdate()
        belief_goodness = list(FRS.belief_state)
        belief_position = list(FRS.belief_position)
        #the end?
        if (FRS.end == True):
            end = 1

    if (total_reward < min_reward):
        min_reward = total_reward
    if (total_reward > max_reward):
        max_reward = total_reward
        best_policy = policy
        best_distance = i
    average_reward += total_reward
    average_distance += i
    success += 1

elapsed_time = time.time() - start_time
print "elapsed_time:", elapsed_time
print "success:", success
print "avg dist:", average_distance/success
print "best dist:", best_distance
print "avg reward:", average_reward/success
print "best reward:", max_reward
print "worst reward:", min_reward
print best_policy
