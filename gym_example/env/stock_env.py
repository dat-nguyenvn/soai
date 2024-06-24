import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pandas as pd
class Stock_env(gym.Env):
    def __init__(self, csv,goal=1500.0,start=1000.0):
        super(Stock_env, self).__init__()
        self.data = pd.read_csv(csv)
        print("init here")
        #data = data.sort_values('Date')
        #prit()
        self.time_step=0
        print("data head", self.data.loc[0,["Close"]])
        
        print("data nuppy", np.float32(self.data.loc[0,["Close"]]).shape)
        #print("data nuppy", int(self.data.loc[0,["Close"]]))

        
        #print("data head", self.data.loc[0,["Close"]])
        self.start = start
        self.goal = goal  # Goal position
        self.asset = start #starting position is current posiiton of agent
        self.cash= start
        self.stock=0.0
        #self.state=self.data.head(5)[["Close"]].to_numpy().reshape(-1)
        self.state=self.data.loc[0:20,["Close"]].to_numpy().reshape(-1)  #21 value
        self.state=np.append(np.array([0]),self.state)
        print("self.state",self.state)
        


        # 4 possible actions: 0=nothing, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)  
        print("self.action_space ",self.action_space )

        # Observation space is grid of size:rows x columns
        #self.observation_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))
        self.observation_space = spaces.Discrete(22) 
    def reset(self):
        print("start episode")
        self.time_step=0
        self.asset = self.start #starting position is current posiiton of agent
        self.cash= self.start
        self.stock=0
        #self.state=self.data.head(5)[["Close"]].to_numpy().reshape(-1)
        self.state=self.data.loc[0:20,["Close"]].to_numpy().reshape(-1)
        self.state=np.append(np.array([0]),self.state)
        #print("self.current_pos",self.current_pos)
        a={"ditmemay": 1}
        return self.state, a # current  array (2,)

    def step(self, action):
        # Move the agent based on the selected action
        #print("self.current_pos +++++++ ---------",self.current_pos)
        #new_balance = self.current
        #print("new_pos +++++++ ---------",new_pos)
        

        #print("action---------------",action)
        #input("Press Enter to continue...")
        truncated=False
        done=False
        
        old_asset=self.asset
        if action == 1:  # buy
            #new_balance = 0
            self.stock=0.995*self.cash/(self.data.loc[self.time_step,["Close"]].to_numpy()[0]) #fee:0.5% buy stock
            self.cash=0.0
            self.asset=self.data.loc[self.time_step,["Close"]].to_numpy()[0] * self.stock
            print("111111111111111",self.asset)
        elif action == 2:  # sell
            #print("haaaaaaaaaaa------------")
            self.cash = self.data.loc[self.time_step,["Close"]].to_numpy()[0]*self.stock*0.995  #fee 0.5% sell/commission
            self.stock=0.0
            self.asset=self.cash
            print("2222222222222",self.asset)
        elif action == 0:  # nothing
            self.cash=self.cash
            self.stock=self.stock
            print("00000000",self.asset)
        

        # check done or not
        #print("type self.cash",type(self.cash))
        #print("self.cash",self.cash)
        #print("self.stock,   ",type(self.stock), "self.stock",self.stock)
        #reward=self.asset
        #reward=1+(self.asset-old_asset)

        if self.cash !=0  or self.stock!=0:
            truncated=False
            reward=1+(self.asset-old_asset)
        else:
            
            #self.asset=self.asset-500
            truncated=True
            reward=-100
            #self.asset=0
            print("eroorrrrr")
        
        if self.asset<100:
            truncated=True
        if self.asset >self.goal:
            
            done =True

        #reward=self.asset
        print("self.asset",self.asset)
        #self.state=self.data.head(5)[["Close"]].to_numpy().reshape(-1)
        self.time_step +=1

        #reward=self.asset
        action_np=np.array([action])
        
        self.state=self.data.loc[self.time_step:self.time_step+20,["Close"]].to_numpy().reshape(-1)

        self.state=np.append(action_np,self.state)
        #print(" self.state", self.state.shape)
        return self.state, reward, done,truncated, {} #truncated

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        return True

    def render(self):
        norender="fgsg"
        # Clear the screen
        # self.screen.fill((255, 255, 255))  

        # # Draw env elements one cell at a time
        # for row in range(self.num_rows):
        #     for col in range(self.num_cols):
        #         cell_left = col * self.cell_size
        #         cell_top = row * self.cell_size
            
        #         try:
        #             h=4
        #             #print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
        #         except Exception as e:
        #             print('Initial state')

        #         if self.maze[row, col] == '#':  # Obstacle
        #             pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
        #         elif self.maze[row, col] == 'S':  # Starting position
        #             pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
        #         elif self.maze[row, col] == 'G':  # Goal position
        #             #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        #             pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
        #         #print("np.array([row, col]).reshape(-1,1)",np.array([row, col]))
        #         #print("self.current_pos",self.current_pos)
        #         if np.array_equal(np.array(self.current_pos), np.array([row, col])):  # Agent position
        #             #print("aaaaaaaaaaaaaaaaaaaaaaaaa")
        #             pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        # pygame.display.update()  # Update the display