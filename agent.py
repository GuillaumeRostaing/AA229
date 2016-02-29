#Class for the agent

class Agent:
    #init
    def __init__(self):
        #position parameters
        self.current_position = [] #current position of the robot
        self.field_of_view = 1 #field of view of the robot
        self.end = 0

        #type of robot: rover (default) or drone
        self.speed = 1
        self.list_of_actions = []

    def Generate(self,position,speed,actions):
        self.current_position = position
        self.speed = speed
        self.list_of_actions = list(actions)

    def SetPosition(self,position):
        self.current_position = position

    def AddAction(self,action):
        self.list_of_actions.append(action)

    def SetAction(self,action):
        self.list_of_actions = action

    def SetSpeed(self,speed):
        self.speed = speed

    def SetEnd(self,end):
        self.end = end
