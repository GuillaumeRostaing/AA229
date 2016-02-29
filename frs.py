import random
import itertools
import numpy as np
import agent

#Python file were we create our class for POMDP

class FRS:
    #init
    def __init__(self):
        #parameters
        self.size = 0 #size of the grid world
        self.rock = 0 #number of rocks
        self.end = False #reached end

        #lists
        self.list_of_positions = []
        self.list_of_rock_positions = [] #list of rock positions
        self.sampled = [] #visited rocks

        #agents
        self.nb_agents = 0
        self.agents = []

        #rewards
        self.rock_reward = 100
        self.rock_penalty = -20
        self.not_observable = -200
        self.observation_cost = 0
        self.observed_penalty = 0
        self.no_rock_penalty = -100
        self.wall_penaly = -100
        self.exit = 10
        self.coward = -100 #if the rover leaves without sampling the rocks

        #list shift
        self.list_of_rock_goodness = [] #list of rock goodness
        self.belief_state = [] #proba that the rock is good
        self.belief_position = [] #known position of the rocks
        self.tmp_state = [] #temporary belief state for synchronous update
        self.tmp_position = [] #temporary belief rock positions for synchronous update

    #generate
    def Generate(self,size,agents,init_agents_position,init_agents_action,init_rock_position,init_rock_goodness):
        if (len(init_rock_position) != len(init_rock_goodness)):
            print "Impossible to initiate rock sample"

        #grid world size
        self.size = size

        #grid cells
        self.list_of_positions = []
        for i in range(self.size):
            for j in range(self.size):
                self.list_of_positions.append([i,j])

        #rocks
        self.list_of_rock_positions = init_rock_position #list of rock positions
        self.list_of_rock_goodness = init_rock_goodness #list of rock goodness
        self.rock = len(init_rock_goodness)
        self.sampled = [0]*self.rock

        #create agents
        self.nb_agents = len(agents)
        self.agents = [0]*self.nb_agents
        for i in range(self.nb_agents):
            self.agents[i] = agent.Agent()
            self.agents[i].Generate(init_agents_position[i],agents[i],init_agents_action[i])
            #update check action for any rock
            for j in range(self.rock): #one action to check each rock
                self.agents[i].AddAction(j)

        #initial belief for quality
        for i in range(self.rock):
            self.belief_state.append(0.5) #equal proba that rock is good or bad

        #initial belief for positions
        for i in range(self.rock):
            self.belief_position.append(['FOG','FOG']) #no idea where the bloody rocks are

        #temporary beliefs
        self.tmp_state = [[0]]*self.nb_agents
        self.tmp_position = [[0]]*self.nb_agents
        for i in range(self.nb_agents):
            self.tmp_state[i] = list(self.belief_state)
            self.tmp_position[i] = list(self.belief_position)

    #take single action at a time
    def TakeActionSolo(self,action,agent_id,act_spd):
        #check agent id, if speed > 1 take 2 actions
        reward = 0
        observation = []
        for a in action: #an action can be a policy
            result = self.Update(a,agent_id,act_spd)
            reward += result[1]
            observation.append(result[0])

        #the end?
        if sum([self.agents[i].end for i in range(self.nb_agents)]) == self.nb_agents:
            self.end = True

        return [observation,reward]

    #take action, an action can be a policy
    def TakeAction(self,action,agent_id):
        #check agent id, if speed > 1 take 2 actions
        reward = 0
        observation = []
        act_spd = 0
        for a in action: #an action can be a policy
            result = self.Update(a,agent_id,act_spd)
            reward += result[1]
            observation.append(result[0])
            act_spd += 1

        #the end?
        if sum([self.agents[i].end for i in range(self.nb_agents)]) == self.nb_agents:
            self.end = True

        return [observation,reward]

    #get field of view
    def GetFOV(self,position):
        fov = []
        for i in range(-1,2):
            for j in range(-1,2):
                cell = [position[0]+i,position[1]+j]
                if (cell in self.list_of_positions):
                    fov.append(cell)
        #check if rocks are in field of view
        present_rock = []
        is_rock = False
        for i in self.list_of_rock_positions: #check if rocks in field of view
            if (i in fov):
                present_rock.append(i)
                is_rock = True
        #return bool (is rock in field of view?), list of rock presents, field of view
        return [is_rock,present_rock,fov]

    #result of action
    def Update(self,action,agent_id,act_spd):
        current_position = self.agents[agent_id].current_position
        if act_spd == 0:
            self.tmp_state[agent_id] = list(self.belief_state)
            self.tmp_position[agent_id] = list(self.belief_position)

        #set reward and observation
        reward = 0
        observation = "N" #set observation to none

        if self.agents[agent_id].end == 0:
            #up
            if (action == "U"):
                #bounces a wall
                if (current_position[1] == self.size-1):
                    reward = self.wall_penaly
                #else update position
                else:
                    self.agents[agent_id].SetPosition([current_position[0],current_position[1]+1])
                    reward = 0

            #down
            if (action == "D"):
                #bounces a wall
                if (current_position[1] == 0):
                    reward = self.wall_penaly
                #else update position
                else:
                    self.agents[agent_id].SetPosition([current_position[0],current_position[1]-1])
                    reward = 0

            #left
            if (action == "L"):
                #bounces a wall
                if (current_position[0] == 0):
                    reward = self.wall_penaly
                #else update position
                else:
                    self.agents[agent_id].SetPosition([current_position[0]-1,current_position[1]])
                    reward = 0

            #right
            if (action == "R"):
                #reaches terminal state
                if (current_position[0] == self.size - 1) and self.agents[agent_id].end == 0:
                    reward = self.exit
                    self.agents[agent_id].SetEnd(1)

                    #if last agent to leave, check if all rock sampled
                    if sum([self.agents[i].end for i in range(self.nb_agents)]) == self.nb_agents:
                        for i in range(self.rock):
                            if any(bgood == 0.5 for bgood in self.tmp_state[agent_id]) or \
                            (((self.sampled[i] == 0) and (self.list_of_rock_goodness[i] == 1))):
                                reward = self.coward

                #else update position
                else:
                    self.agents[agent_id].SetPosition([current_position[0]+1,current_position[1]])
                    reward = 0


            if (action in ["U","D","L","R"]):
                fov = self.GetFOV(current_position) #returns a bool and the list of rocks in the FOV if bool=true
                if (fov[0] == True):#and (any(item == ['FOG','FOG'] for item in self.belief_position)):
                    observation = 'P' #update observation -> presence of rock
                    for rock_pos in fov[1]:
                        self.tmp_position[agent_id][self.list_of_rock_positions.index(rock_pos)] = rock_pos #update belief

            #sample rock
            if (action == "S"):
                #no rock impossible to sample
                if (current_position not in self.list_of_rock_positions):
                    reward = self.no_rock_penalty
                elif (self.sampled[self.list_of_rock_positions.index(current_position)] == 1):
                    reward = self.rock_penalty
                elif self.tmp_state[agent_id][self.list_of_rock_positions.index(current_position)] == 0.5:
                    rock_id = self.list_of_rock_positions.index(current_position)
                    self.sampled[rock_id] = 1
                    self.tmp_state[agent_id][rock_id] = 0
                    reward = self.no_rock_penalty
                    #bad rock
                    if (self.list_of_rock_goodness[rock_id] == 0):
                        observation = 0
                    #good rock
                    else:
                        self.list_of_rock_goodness[rock_id] = 0 #switch good rock to bad
                        observation = 1
                #rock is present
                else:
                    rock_id = self.list_of_rock_positions.index(current_position)
                    self.sampled[rock_id] = 1
                    self.tmp_state[agent_id][rock_id] = 0 #whatever the rock is, now we're sure it will be bad
                    #bad rock
                    if (self.list_of_rock_goodness[rock_id] == 0):
                        reward = self.rock_penalty
                        observation = 0
                    #good rock
                    else:
                        self.list_of_rock_goodness[rock_id] = 0 #switch good rock to bad
                        reward = self.rock_reward
                        observation = 1

            #check rock
            if isinstance(action,int) == True:
                rock_id = action
                reward = self.observation_cost

                #check if rock in field of view
                fov = self.GetFOV(current_position)

                if (self.list_of_rock_positions[rock_id] not in fov[2]):
                    reward = self.not_observable
                elif self.tmp_state[agent_id][rock_id] != 0.5:
                    reward = self.observed_penalty
                else:
                    old_belief = self.tmp_state[agent_id][rock_id]
                    if (self.list_of_rock_positions[rock_id] == current_position): #return goodness
                        observation = self.list_of_rock_goodness[rock_id]
                        self.tmp_state[agent_id][rock_id] = observation #update belief
                    else:
                        proba = .9
                        if (random.random() < proba):
                            observation = self.list_of_rock_goodness[rock_id]
                        else:
                            observation = 1-self.list_of_rock_goodness[rock_id]
                        #update belief
                        if (observation == 1):
                            self.tmp_state[agent_id][rock_id] = 1
                        else:
                            self.tmp_state[agent_id][rock_id] = 0

        return [observation,reward]

    def BeliefUpdate(self):
        for j in range(len(self.belief_state)):
            change = self.belief_state[j]
            for i in range(self.nb_agents):
                if self.belief_state[j] != self.tmp_state[i][j]:
                    change = self.tmp_state[i][j]
            self.belief_state[j] = change
        for j in range(len(self.belief_position)):
            change = self.belief_position[j]
            for i in range(self.nb_agents):
                if self.belief_position[j] != self.tmp_position[i][j]:
                    change = self.tmp_position[i][j]
            self.belief_position[j] = change
