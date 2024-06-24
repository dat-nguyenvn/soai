import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class DynamicEnv(gym.Env):
    def __init__(self, maze):
        super(DynamicEnv, self).__init__()
        #self.maze = np.array(maze)  # Maze represented as a 2D numpy array
        self.path_frames=''
        
        self.start_posi = np.where(self.maze == 'S')  # Starting position
        self.goal_posi = np.where(self.maze == 'G')  # Goal position
        
        self.current_pos = self.start_posi #starting position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)  

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize Pygame
        pygame.init()
        self.cell_size = 125

        # setting display size
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self):
        #print("haha")
        #print("self.start_pos",self.start_pos)
        self.start_pos=np.array([self.start_posi[0][0],self.start_posi[1][0]])
        #print(" shape self.start_pos",self.start_pos.shape)
        self.goal_pos=np.array([self.goal_posi[0][0],self.goal_posi[1][0]])


        self.current_pos = self.start_pos #goc
        #print("self.current_pos",self.current_pos)
        a={"ditmemay": 1}
        return self.current_pos, a # current  array (2,)

    def step(self, action):
        # Move the agent based on the selected action
        #print("self.current_pos +++++++ ---------",self.current_pos)
        new_pos = np.array(self.current_pos)
        #print("new_pos +++++++ ---------",new_pos)


        #print("action---------------",action)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            #print("haaaaaaaaaaa------------")
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
            #print("self.current_pos updateeeeeeeeeeeeeeeeeeeeeee",self.current_pos)
        # else:
        #     done=True
        #     print("die")
        # Reward function
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 1.0
            done = True
            print("win")
        else:
            reward = 0.0
            done = False

        return self.current_pos, reward, done, {}

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
        # Clear the screen
        self.screen.fill((255, 255, 255))  

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
                try:
                    h=4
                    #print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')

                if self.maze[row, col] == '#':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Goal position
                    #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                #print("np.array([row, col]).reshape(-1,1)",np.array([row, col]))
                #print("self.current_pos",self.current_pos)
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):  # Agent position
                    #print("aaaaaaaaaaaaaaaaaaaaaaaaa")
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display