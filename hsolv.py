#Heuristics Solver

import frs
import math

class HSolv:
    #init
    def __init__(self):
        #parameters
        self.FRS = None
        self.observed_states = []
        self.gamma = 0.9
        self.cumulative_reward = 0
        self.iter = 0

        #lists
        self.seen = [] #states seen by the agents
        self.sampled = [] #sampled rockss
        self.to_sample = [] #rocks to sample
        self.checked = [] #checked rocks
        self.to_check = [] #rocks to check
        self.end = False #end of simulation

    def Generate(self,grid_size,agents,init_agents_position,init_agents_action,rock_positions,rock_goodness,gamma_):
        #update parameters
        self.FRS = frs.FRS()
        self.FRS.Generate(grid_size,agents,init_agents_position,init_agents_action,rock_positions,rock_goodness)
        self.gamma = gamma_

    def Solve(self):
        #agents act
        while self.end == False and self.iter < 30:
            for agent_id in range(self.FRS.nb_agents):
                print "agent-action"
                print agent_id
                for i in range(self.FRS.agents[agent_id].speed):
                    action = [self.SelectAction(agent_id)]
                    result = self.FRS.TakeActionSolo(action,agent_id,i)
                    print action
                    print self.FRS.agents[agent_id].current_position
                    self.cumulative_reward += math.pow(self.gamma,self.iter)*result[1]
            self.FRS.BeliefUpdate()
            self.iter += 1
            if sum([self.FRS.agents[i].end for i in range(self.FRS.nb_agents)]) == self.FRS.nb_agents:
                self.end = True

    def SelectAction(self,agent_id):
        #when all rocks believed to be bad, leave
        if sum(self.FRS.belief_state) == 0:
            return "R"

        #check field of view and update belief
        current_position = self.FRS.agents[agent_id].current_position
        self.UpdateFOV()

        #anything to check?
        for elem in self.to_check:
            if abs(current_position[0]-elem[0]) < 2 and abs(current_position[1]-elem[1]) < 2:
                self.to_check.remove(elem)
                self.checked.append(elem)
                return self.FRS.list_of_rock_positions.index(elem)

        #update list of rocks to sample
        for i in range(len(self.FRS.belief_state)):
            if self.FRS.belief_state[i] > 0.5 and self.FRS.list_of_rock_positions[i] not in self.to_sample:
                self.to_sample.append(self.FRS.list_of_rock_positions[i])

        if "S" in self.FRS.agents[agent_id].list_of_actions:
            min_dist = 2*math.pow(self.FRS.size,2)
            target = 0
            for elem in self.to_sample:
                dist = (abs(current_position[0]-elem[0])+ abs(current_position[1]-elem[1]))
                if dist == 0:
                    self.to_sample.remove(elem)
                    self.sampled.append(elem)
                    return "S"
                else:
                    if dist < min_dist:
                        min_dist = dist
                        target = elem
            if (target != 0):
                if current_position[0] > target[0]:
                    return "L"
                if current_position[0] < target[0]:
                    return "R"
                if current_position[1] > target[1]:
                    return "D"
                if current_position[1] < target[1]:
                    return "U"

        #if nothing to check/sample, go to unseen zones
        for i in range(self.FRS.size):
            for j in range(self.FRS.size):
                if [i,j] not in self.seen:
                    if current_position[0] > i:
                        return "L"
                    if current_position[0] < i:
                        return "R"
                    if current_position[1] > j:
                        return "D"
                    if current_position[1] < j:
                        return "U"

    def UpdateFOV(self):
        for agent_id in range(self.FRS.nb_agents):
            current_position = self.FRS.agents[agent_id].current_position
            fov = self.FRS.GetFOV(current_position)
            if (fov[0] == True):#and (any(item == ['FOG','FOG'] for item in self.belief_position)):
                for rock_pos in fov[1]:
                    if self.FRS.belief_position[self.FRS.list_of_rock_positions.index(rock_pos)] == ['FOG','FOG']:
                        self.FRS.belief_position[self.FRS.list_of_rock_positions.index(rock_pos)] = rock_pos #update belief
                        self.to_check.append(rock_pos)
            for elem in fov[2]:
                if elem not in self.seen:
                    self.seen.append(elem)
