import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]

class  SingleAgentThreeDGridEnv:
    def __init__(self, stay_penalty=-2, collision_penalty=-2, goal_reward=10, step_penalty=-1):
        self.move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
        self.shape = (10, 10, 10)
        self.start = (0, 0, 0)
        self.n_obstacles = 15
        self.stay_penalty = stay_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.Q = np.zeros((1000, len(move)))
        self.convergence = 0
        
        self.destination= (9,9,9) #tuple(np.random.randint(0, s) for s in shape)
        
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            obstacle = tuple(np.random.randint(0, s) for s in self.shape)
            if obstacle != self.start and obstacle != self.destination:
                self.obstacles.add(obstacle)
              
        self.agent_position = self.start
        assert self.start not in self.obstacles, "Start position must not be an obstacle"
        assert self.destination not in self.obstacles, "Destination position must not be an obstacle"

    def reset(self):
        self.agent_position = self.start
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            obstacle = tuple(np.random.randint(0, s) for s in self.shape)
            if obstacle != self.start and obstacle != self.destination:
                self.obstacles.add(obstacle)
        return self._get_state()
    
    def start_state(self):
        self.agent_position = self.start
        return self._get_state()

    def step(self, action):
        
        new_position = tuple(np.add(self.agent_position, self.move[action]))

        if new_position != self.agent_position:
            if (0 <= new_position[0] < self.shape[0] and
                0 <= new_position[1] < self.shape[1] and
                0 <= new_position[2] < self.shape[2] and
                new_position not in self.obstacles):
                self.agent_position = new_position
                reward = self.goal_reward if new_position == self.destination else self.step_penalty
                done = new_position == self.destination
            else:
                reward = self.collision_penalty
                done = False
        else:
            reward = self.stay_penalty
            done = False

        return self._get_state(), reward, done

    def q_computing(self):
        action = np.random.randint(low=0, high=len(self.move))
        state_prio = self._get_state()
        state_after, reward, done = self.step(action=action)
        x = np.where(state_prio.flatten() == 1)[0][0]
        y = action
        self.Q[x,y] = reward + (np.max(self.Q[np.where(state_after.flatten()==1)]))
        return done
    def policy_comp(self):
        policy = []
        done =  False
        while done == False:
            state_prio = self._get_state()
            x = np.where(state_prio.flatten() == 1)[0][0]
            y = np.argmax(self.Q[x])
            policy.append(y)
            state, reward, done = self.step(action=y)
        return policy

    def train(self):
        done = False
        prev_q = self.Q
        while not done:
            done = self.q_computing()
        self.convergence = np.abs(np.sum(prev_q) - np.sum(self.Q))
        self.agent_position = self.start



    def _get_state(self):
        state = np.zeros(self.shape)
        for obs in self.obstacles:
            state[obs] = -1
        state[self.destination] = -2
        state[self.agent_position] = 1
        return state

    # You can modify or ignore the following (see also step())   
    def _get_reward(self, position):
        if position == self.destination:  
            return 100
        return -1

    def render(self):
      self.ax.cla()  # Clear the axis to redraw

      grid = self._get_state()  # Get the grid state

      # Only fill spaces that are not empty
      filled = np.zeros(grid.shape, dtype=bool)
      filled[grid != 0] = True

      # Create a color mapping for the grid
      colors = np.empty(grid.shape, dtype=object)
      colors[grid == -1] = 'red'    # Obstacles in red
      colors[grid == 1] = 'green'   # Agent in green
      colors[grid == -2] = 'yellow' # Destination in yellow

      # Plot the grid using voxels
      self.ax.voxels(filled, facecolors=colors, edgecolor='k')

      self.ax.set_xlabel('X axis')
      self.ax.set_ylabel('Y axis')
      self.ax.set_zlabel('Z axis')

    def animate_solution(self, policy):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      self.ax = ax

      # Prepare the animation function
      def update(num):
          action = policy[num]
          self.step(action)
          self.render()
          return self.ax,

      # Create the animation
      ani = animation.FuncAnimation(fig, update, frames=len(policy), blit=False, interval=500, repeat=False)
      plt.close(fig)  # Prevent duplicate display
      return ani
    
    """
    Further methods can be defined if necessary (e.g., random_state(), ...)     
    """


class MultiAgentThreeDGridEnv:
    def __init__(self, shape=(10, 10, 10), starts=[(0, 0, 0), (0, 0, 9)], destinations=[(9, 9, 9), (9, 9, 0)], n_obstacles=15, stay_penalty=-15, collision_penalty=-10, goal_reward=100, step_penalty=-1):
        self.shape = shape
        self.starts = starts
        self.destinations = destinations
        self.n_obstacles = n_obstacles
        self.stay_penalty = stay_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        self.obstacles = self._generate_obstacles()
        self.agent_positions = list(starts)
        
        assert not (set(starts) & self.obstacles), "Start positions must not be in obstacles"
        assert not (set(destinations) & self.obstacles), "Destination positions must not be in obstacles"

    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < self.n_obstacles:
            obstacle = tuple(np.random.randint(0, s) for s in self.shape)
            if obstacle not in self.starts and obstacle not in self.destinations:
                obstacles.add(obstacle)
        return obstacles

    def reset(self):
        self.agent_positions = list(self.starts)
        self.obstacles = self._generate_obstacles()
        return [self._get_state(i) for i in range(len(self.starts))]

    def step(self, actions):
        new_positions = []
        rewards = []
        dones = []

        for i, action in enumerate(actions):
            move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
            new_position = tuple(np.add(self.agent_positions[i], move[action]))

            if new_position == self.agent_positions[i]:
                reward = self.stay_penalty
            elif (0 <= new_position[0] < self.shape[0] and
                  0 <= new_position[1] < self.shape[1] and
                  0 <= new_position[2] < self.shape[2] and
                  new_position not in self.obstacles and
                  new_positions.count(new_position) == 1):
                self.agent_positions[i] = new_position
                reward = self.goal_reward if new_position == self.destinations[i] else self.step_penalty
            else:
                reward = self.collision_penalty

            new_positions.append(new_position)
            rewards.append(reward)
            dones.append(new_position == self.destinations[i])

        states = [self._get_state(i) for i in range(len(self.starts))]
        return states, rewards, dones

    def _get_state(self, agent_index):
        state = np.zeros(self.shape)
        for obs in self.obstacles:
            state[obs] = -1
        state[self.destinations[agent_index]] = -2
        for i, pos in enumerate(self.agent_positions):
            if state[pos] == 0:
                state[pos] = i + 1
        return state

    def _get_reward(self, agent_index, position):
        if position == self.destinations[agent_index]:
            return self.goal_reward
        return self.step_penalty
    
    """
    Further methods can be defined if necessary (e.g., random_state(), start_state(), animation, ...)     
    """
def system_description(agent):
    print(f"Q matrix: {agent.Q}")
    print(f"Starting point: {agent.start}")

def policy_show(agent):
    print(single_agent.policy_comp())
single_agent = SingleAgentThreeDGridEnv()
single_agent.start_state()
for i in range(100):
    single_agent.train()
#policy_show(single_agent)
system_description(single_agent)

