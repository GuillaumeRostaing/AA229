#MCTS
import operator
import numpy as np
import frs
import random
from random import randint
import agent

class Rollouts:

    #init class
    def __init__(self):
        #hyperparameters
        self.gamma = 0.95 #discount factor

        #rock sample parameters and instances
        self.FRS = None #rock sample
        self.agents = [] #list of agents' speeds
        self.init_agents_position = [] #list of agents' positions
        self.init_agents_action = [] #list of initial agents' actions
        self.agents_action = [] #list of agents' actions
        self.n = 0 #size grid world
        self.nb_rocks = 0 #nb of rocks
        self.belief = []

    #initiate important parameters
    def InitiateParameters(self,n_,agents,init_agents_position,init_agents_action,agents_action,nb_rocks,gamma_):
        self.n = n_
        self.agents = agents
        self.init_agents_position = init_agents_position
        self.init_agents_action = init_agents_action
        self.agents_action = agents_action
        self.nb_rocks = nb_rocks
        self.gamma = gamma_

    #select action
    def SelectAction(self,b_goodness,b_position,pos,d,nb_iter,factored,selection):
        self.belief = b_goodness

        tabu = [] #tabu action
        for i in range(len(b_goodness)):
            if b_goodness[i] != 0.5:
                tabu.append(i)

        for agent_id in range(len(self.agents)):
            a = []
            for action in self.agents_action[agent_id]:
                if action not in tabu:
                    a.append(action)
            self.agents_action[agent_id] = list(a)

        if factored == True:
            joint_action = []
            for agent_id in range(len(self.agents)):

                if self.agents[agent_id] == 1:
                    Q = {}
                    for action1 in self.agents_action[agent_id]:
                        action = [action1]
                        Q[tuple(action)] = 0
                        for i in range(nb_iter): #sample
                            s =  self.Draw(pos,b_goodness,b_position)
                            self.SimulatorInit(s)
                            [s_n,o,r] = self.SimulatorGenerate(s,action,agent_id)
                            r += self.Rollout(s,d,selection,agent_id)
                            Q[tuple(action)] += r
                        Q[tuple(action)] = float(Q[tuple(action)])/nb_iter
                    value_sorted = sorted(Q, key=Q.get, reverse=True)
                    joint_action.append(list(value_sorted[0]))


                if self.agents[agent_id] == 2:
                    Q = {}
                    for action1 in self.agents_action[agent_id]:
                        for action2 in self.agents_action[agent_id]:
                            action = [action1,action2]
                            Q[tuple(action)] = 0
                            for i in range(nb_iter): #sample
                                s =  self.Draw(pos,b_goodness,b_position)
                                self.SimulatorInit(s)
                                [s_n,o,r] = self.SimulatorGenerate(s,action,agent_id)
                                r += self.Rollout(s,d,'random',agent_id)
                                Q[tuple(action)] += r
                    value_sorted = sorted(Q, key=Q.get, reverse=True)
                    joint_action.append(list(value_sorted[0]))
            return joint_action

    #rollout
    def Rollout(self,s,d,policy,agent_id):
        if (d == 0):
            return 0
        action = self.Policy(s,policy,agent_id) #chose action according to "policy"
        [s_n,o,r] = self.SimulatorGenerate(s,action,agent_id) #generate
        if (self.FRS.end == True) or r > 0:
            d = 1
            self.FRS.end = False
        return self.gamma*(r + self.gamma*self.Rollout(s_n,d-1,policy,agent_id))


    #draw sample
    def Draw(self,pos,b_goodness,b_position):
        state = [position for position in pos] #agent's positions
        rock_position = []
        for i in b_position:
            if (i == ['FOG','FOG']):
                b = [randint(0,self.n-1),randint(0,self.n-1)]
                while (b in rock_position) and (b not in b_position):
                    b = [randint(0,self.n-1),randint(0,self.n-1)]
                rock_position.append(b)
                state.append(b)
            else:
                state.append(i)
        for i in range(len(b_goodness)):
            if (random.random() < b_goodness[i]):
                state.append(1)
            else:
                state.append(0)
        return state

    #initiate simulator
    def SimulatorInit(self,s):
        #rock goodness comes from the state s
        init_rock_position = [s[i] for i in range(len(self.agents),self.nb_rocks+len(self.agents))]
        init_rock_goodness = [s[i] for i in range(len(self.agents)+self.nb_rocks,len(s))]

        #initiate rock sample
        self.FRS = frs.FRS()

        #generate rock sample
        self.FRS.Generate(self.n,self.agents,self.init_agents_position,self.init_agents_action,init_rock_position,init_rock_goodness)
        self.FRS.belief_state = list(self.belief)

    def SimulatorGenerate(self,s,a,agent_id):
        s_n = []
        observation = []
        reward = 0
        action = a
        if isinstance(a,int) == True:
            action = [a]
        sample = self.FRS.TakeAction(action,agent_id)
        observation.append(tuple(sample[0]))
        reward += sample[1]
        self.FRS.BeliefUpdate()
        for agent in self.FRS.agents:
            s_n.append(agent.current_position)

        [s_n.append(s[i]) for i in range(len(self.agents),len(s))]
        return [s_n,observation,reward]

    def Policy(self,s,method,agent_id):
        if (method == 'random'):
            p = []
            for i in range(self.agents[agent_id]): #draw 'speed' random actions per agent
                p.append(random.choice(self.agents_action[agent_id]))
            return p

        if (method == 'legal'):
            illegal = []
            #if robot on the edge
            if (s[agent_id][0] == 0): #if on the left side
                illegal.append("L")
            if (s[agent_id][1] == 0): #if on the bottom
                illegal.append("D")
            if (s[agent_id][1] == self.n-1): #if at the top
                illegal.append("U")
            if (s[agent_id][0] == self.n-1) and any(item > 0.5 for item in self.FRS.belief_state): #if at the right side
                illegal.append("R")
            if (s[agent_id] not in s[len(self.agents):self.nb_rocks+len(self.agents)]): #if robot not on a rock don't sample
                illegal.append("S")
            if (s[agent_id] in s[len(self.agents):self.nb_rocks+len(self.agents)]): #if robot on a rock
                if (s[s[len(self.agents):].index(s[agent_id])+self.nb_rocks+len(self.agents)] <= 0.5): #if we believe rock is bad don't sample
                    illegal.append("S")
            for i in range(0,self.nb_rocks): #if rock already sampled don't sample it again
                if self.FRS.belief_state[i] != 0.5:
                    illegal.append(i)

            #create set of legal actions
            set_of_actions = []
            for action in self.agents_action[agent_id]:
                if action not in illegal:
                    set_of_actions.append(action)

            a = []
            #if all rocks bad, go right
            if all(item < 0.5 for item in self.FRS.belief_state):
                for i in range(self.agents[agent_id]):
                    a.append("R")
            elif (s[agent_id] in s[len(self.agents):self.nb_rocks+len(self.agents)]): #if robot on a rock
                if (s[s[len(self.agents):].index(s[agent_id])+self.nb_rocks+len(self.agents)] > 0.5):
                    if "S" in self.agents_action[agent_id]:
                        a.append("S")
                        if self.agents[agent_id] > 1:
                            for i in range(self.agents[agent_id])-1:
                                a.append(random.choice(set_of_actions))
            else: #pick actions at random
                for i in range(self.agents[agent_id]):
                    a.append(random.choice(set_of_actions))
            if not a:
                for i in range(self.agents[agent_id]):
                    a.append(random.choice(set_of_actions))
            return a
