#MCTS
import operator
import numpy as np
import frs
import random
from random import randint
import agent

class FSMCTS:

    #init class
    def __init__(self):
        #lists simulation
        self.Q = {} #Q values
        self.N = {} #count values
        self.T = [] #visit list

        #hyperparameters
        self.c = 0.8 #exploration parameter
        self.gamma = 0.9 #discount factor

        #rock sample parameters and instances
        self.FRS = None #rock sample
        self.FRSr = None #rock sample rollout
        self.agents = [] #list of agents' speeds
        self.init_agents_position = [] #list of agents' positions
        self.init_agents_action = [] #list of initial agents' actions
        self.agents_action = [] #list of agents' actions
        self.n = 0 #size grid world
        self.nb_rocks = 0 #nb of rocks
        self.belief = []


    #initiate important parameters
    def InitiateParameters(self,n_,agents,init_agents_position,init_agents_action,agents_action,nb_rocks,c_,gamma_):
        self.n = n_
        self.agents = agents
        self.init_agents_position = init_agents_position
        self.init_agents_action = init_agents_action
        self.agents_action = agents_action
        self.nb_rocks = nb_rocks
        self.c = c_
        self.gamma = gamma_

    #select action
    def SelectAction(self,b_goodness,b_position,pos,d,nb_iter,selection):
        h = ["empty"] #initial empty history
        self.belief = b_goodness

        #reinitialize
        self.Q = {} #Q values
        self.N = {} #count values
        self.T = [] #visit list

        #sample
        for i in range(nb_iter):
            s =  self.Draw(pos,b_goodness,b_position)
            self.Simulate(s,h,d,selection)
            self.SimulatorInit(s,'simulate')

        joint_action = []
        for i in range(len(self.agents)):
            max_value = -10000
            if self.agents[i] == 1: #only 2 speeds currently supported
                best_action = None
                for action in self.agents_action[i]:
                    if self.Q[i,('empty',tuple([action]))] > max_value:
                        max_value = self.Q[i,('empty',tuple([action]))]
                        best_action = action
                joint_action.append(best_action)
            elif self.agents[i] == 2:
                best_action = None
                for action1 in self.agents_action[i]:
                    for action2 in self.agents_action[i]:
                        action = (action1,action2)
                        if self.Q[i,('empty',action)] > max_value:
                            max_value = self.Q[i,('empty',action)]
                            best_action = action
                joint_action.append(list(best_action))
        return joint_action


    #simulate
    def Simulate(self,s,h,d,selection):
        #base case
        if (d == 0):
            return 0

        #expansion and rollout
        if (h not in self.T):
            for i in range(len(self.agents)): #factored on each agent
                if self.agents[i] == 1: #currently only 2 speeds possible, 1 or 2
                    for action in self.agents_action[i]:
                        h_local = list(h) #copy list h
                        h_local.append(tuple([action]))
                        self.N[i,tuple(h_local)] = 1 #initiate to 1
                        self.Q[i,tuple(h_local)] = 0
                elif self.agents[i] == 2:
                    for action1 in self.agents_action[i]:
                        for action2 in self.agents_action[i]:
                            h_local = list(h) #copy list h
                            action = (action1,action2)
                            h_local.append(action)
                            self.N[i,tuple(h_local)] = 1 #initiate to 1
                            self.Q[i,tuple(h_local)] = 0
            self.T.append(h)
            self.SimulatorInit(s,'rollout')
            return self.Rollout(s,h,d,selection)

        #search
        #get joint action
        joint_action = []
        for i in range(len(self.agents)):
            if self.agents[i] == 1:
                sum_N = 0
                for action in self.agents_action[i]:
                    h_local = list(h)
                    h_local.append(tuple([action]))
                    sum_N += self.N[i,tuple(h_local)]
                value = -10000
                a_max = None
                for action in self.agents_action[i]:
                    h_local = list(h)
                    h_local.append(tuple([action]))
                    local_v = self.Q[i,tuple(h_local)] + self.c*np.sqrt(np.log(sum_N)/self.N[i,tuple(h_local)])
                    if local_v > value:
                        value = local_v
                        a_max = action
                joint_action.append([a_max])
            elif self.agents[i] == 2:
                sum_N = 0
                for action1 in self.agents_action[i]:
                    for action2 in self.agents_action[i]:
                        h_local = list(h)
                        action = (action1,action2)
                        h_local.append(action)
                        sum_N += self.N[i,tuple(h_local)]
                value = -10000
                a_max = None
                for action1 in self.agents_action[i]:
                    for action2 in self.agents_action[i]:
                        h_local = list(h)
                        action = (action1,action2)
                        h_local.append(action)
                        local_v = self.Q[i,tuple(h_local)] + self.c*np.sqrt(np.log(sum_N)/self.N[i,tuple(h_local)])
                        if local_v > value:
                            value = local_v
                            a_max = action
                joint_action.append(list(a_max))

        #generate
        [s_n,o,r] = self.SimulatorGenerate(s,joint_action,'simulate')
        hp = list(h) #copy list h
        history_action = []
        for action in joint_action:
            history_action.append(tuple(action))
        hp.append(tuple(history_action))
        hp.append(tuple(o))
        if (self.FRS.end == True):
            d = 1
        q = r + self.gamma*self.Simulate(s_n,hp,d-1,selection)
        h_local = list(h)
        for i in range(len(self.agents)):
            h_agent = list(h_local)
            h_agent.append(tuple(history_action[i]))
            self.N[i,tuple(h_agent)] += 1
            self.Q[i,tuple(h_agent)] += (q - self.Q[i,tuple(h_agent)])/self.N[i,tuple(h_agent)]
    #    print d
    #    print hp
    #    print tuple(h_local)
    #    print self.Q[tuple(h_local)]
        return q


    #rollout
    def Rollout(self,s,h,d,selection):
        if (d == 0):
            return 0
        action = self.Policy(s,selection) #chose action
        [s_n,o,r] = self.SimulatorGenerate(s,action,'rollout') #generate
        h_local = list(h) #copy list h
        h_local.append(tuple(action))
        h_local.append(tuple(o))
        if (self.FRSr.end == True):
            d = 1
            self.FRSr.end = False
        return (r + self.gamma*self.Rollout(s_n,h_local,d-1,selection))


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
    def SimulatorInit(self,s,fct):
        #rock goodness comes from the state s
        init_rock_position = [s[i] for i in range(len(self.agents),self.nb_rocks+len(self.agents))]
        init_rock_goodness = [s[i] for i in range(len(self.agents)+self.nb_rocks,len(s))]

        if (fct == 'simulate'):
            #initiate rock sample
            self.FRS = frs.FRS()

            #generate rock sample
            self.FRS.Generate(self.n,self.agents,self.init_agents_position,self.init_agents_action,init_rock_position,init_rock_goodness)
            self.FRS.belief_state = list(self.belief)

        else:
            #initiate rock sample
            self.FRSr = frs.FRS()

            #generate rock sample
            self.FRSr.Generate(self.n,self.agents,self.init_agents_position,self.init_agents_action,init_rock_position,init_rock_goodness)
            self.FRSr.belief_state = list(self.belief)

    def SimulatorGenerate(self,s,a,fct):
        s_n = []
        observation = []
        reward = 0
        if (fct == 'simulate'):
            for agent_id in range(self.FRS.nb_agents):
                action = a[agent_id]
                if isinstance(action,int) == True:
                    action = [action]
                sample = self.FRS.TakeAction(action,agent_id)
                observation.append(tuple(sample[0]))
                reward += sample[1]
            self.FRS.BeliefUpdate()
            for agent in self.FRS.agents:
                s_n.append(agent.current_position)
        else:
            for agent_id in range(self.FRSr.nb_agents):
                action = a[agent_id]
                if isinstance(action,int) == True:
                    action = [action]
                sample = self.FRSr.TakeAction(action,agent_id)
                observation.append(tuple(sample[0]))
                reward += sample[1]
            self.FRSr.BeliefUpdate()
            for agent in self.FRSr.agents:
                s_n.append(agent.current_position)

        [s_n.append(s[i]) for i in range(len(self.agents),len(s))]
        return [s_n,observation,reward]


    def Policy(self,s,method):
        if (method == 'random'):
            p = []
            for i in range(len(self.agents)): #agent id
                action = []
                for j in range(self.agents[i]): #draw 'speed' random actions per agent
                    action.append(random.choice(self.agents_action[i]))
                p.append(action)
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
            #print s[0]
            #print a
            return a
