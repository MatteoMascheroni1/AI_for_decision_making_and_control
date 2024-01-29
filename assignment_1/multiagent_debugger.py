import numpy as np
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
        self.alpha = 1
        self.obstacles = self._generate_obstacles()
        self.agent_positions = list(starts)
        # initialise Q matrixes
        self.Q = [np.zeros((1000,7)),np.zeros((1000,7))]

        assert not (set(starts) & self.obstacles), "Start positions must not be in obstacles"
        assert not (set(destinations) & self.obstacles), "Destination positions must not be in obstacles"
        assert not(set([(0,0,1),(0,1,0),(1,0,0)])& self.obstacles), "Agent 0 is blocked"
        assert not(set([(0,0,8),(0,1,9),(1,0,9)])& self.obstacles), "Agent 1 is blocked"
        assert not(set([(9,9,8),(8,9,9),(9,8,9)])& self.obstacles), "Destination 0 is blocked"
        assert not(set([(9,9,1),(8,9,0),(9,8,0)])& self.obstacles), "Destination 0 is blocked"

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

    def restart(self):
        self.agent_positions = list(self.starts)
        return [self._get_state(i) for i in range(len(self.starts))]

    def select_action(self, epsilon=1):
        actions = []
        move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
        # epsilon is the probability of choosing a random action
        if np.random.binomial(1, epsilon):
             actions = [np.random.randint(len(move)), np.random.randint(len(move))]
        # perform the best action
        else:
            prev_state = [self._get_state_single(0), self._get_state_single(1)]
            for i, state in enumerate(prev_state):
                actions.append(np.argmax(self.Q[i][np.where(state.flatten() == i+1)]))
        return actions


    def step(self, actions):
        new_positions = []
        rewards = []
        dones = []

        for i, action in enumerate(actions):
            move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
            # added that if the agent is in the destination it will wait there and take no reward
            if self.agent_positions[i] == self.destinations[i]:
                print(f"Agent {i} is arrived")
                dones.append(True)
                rewards.append(0)
                new_positions.append(self.destinations[i])
            else:
                new_position = tuple(np.add(self.agent_positions[i], move[action]))
                if new_position == self.agent_positions[i]:
                    reward = self.stay_penalty
                elif (0 <= new_position[0] < self.shape[0] and
                      0 <= new_position[1] < self.shape[1] and
                      0 <= new_position[2] < self.shape[2] and
                      new_position not in self.obstacles and
                      new_positions.count(new_position) != 2):
                    # given the check on the conditions the two agents could collide only if they move in the same cell
                    self.agent_positions[i] = new_position
                    reward = self.goal_reward if new_position == self.destinations[i] else self.step_penalty
                else:
                    reward = self.collision_penalty
                    new_position = self.agent_positions[i]

                new_positions.append(new_position)
                rewards.append(reward)
                dones.append(new_position == self.destinations[i])

        states = [self._get_state_single(i) for i in range(len(self.starts))]
        return states, rewards, dones

    def episode(self, epsilon=1):
        self.agent_positions = list(self.starts)
        dones = [False * 2]
        steps = 0
        while dones.count(True) != 2:
            steps += 1
            actions = self.select_action(epsilon=epsilon)
            prev_state = [self._get_state_single(0), self._get_state_single(1)]
            new_state,rewards, done = self.step(actions=actions)
            dones = done
            # Q update
            for i,pos in enumerate([1,2]):
                x = np.where(prev_state[i].flatten() == pos)[0][0]
                y = actions[i]
                self.Q[i][x,y] = self.Q[i][x,y] + self.alpha * (rewards[i] + np.max(self.Q[i][np.where(new_state[i].flatten() == pos)]) - self.Q[i][x,y])
            print(f"Number of steps: {steps}")

    def value_episode(self, actions):
        self.agent_positions = list(self.starts)
        dones = [False * 2]
        reward_0 = []
        reward_1 = []
        while dones.count(True) != 2:
            actions = [np.random.randint(7), np.random.randint(7)]
            new_state,rewards, done = self.step(actions=actions)
            reward_0.append(rewards[0])
            reward_1.append(rewards[1])
            dones = done
        return reward_0.sum(), reward_1.sum()


    def _get_state(self, agent_index):
        state = np.zeros(self.shape)
        for obs in self.obstacles:
            state[obs] = -1
        state[self.destinations[agent_index]] = -2
        for i, pos in enumerate(self.agent_positions):
#            if state[pos] == 0:
            state[pos] = i + 1
        return state

    def _get_state_render(self):
        state = np.zeros(self.shape)
        for obs in self.obstacles:
            state[obs] = -1
        state[self.destinations[0]] = -2
        state[self.destinations[1]] = -3
        for i, pos in enumerate(self.agent_positions):
            state[pos] = i+1
        return state

    def _get_state_single(self, agent_index):
        state = np.zeros(self.shape)
        for obs in self.obstacles:
            state[obs] = -1
        state[self.destinations[agent_index]] = -2
        state[self.agent_positions[agent_index]] = agent_index + 1
        return state

    def _get_reward(self, agent_index, position):
        if position == self.destinations[agent_index]:
            return self.goal_reward
        return self.step_penalty

    def policy_state(self, agent_index, pos):
        state =  np.zeros(self.shape)
        state[pos] = agent_index + 1
        return state

    def _get_policy(self, agent_index):
        pos = self.starts[agent_index]
        move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
        done = False
        policy = []
        state =  self.policy_state(agent_index,pos)
        while not done:
            s = np.where(state.flatten() == agent_index+1)
            best_action = np.argmax(self.Q[agent_index][s])
            policy.append(best_action)
            pos = tuple(np.add(pos, move[best_action]))
            done = pos == self.destinations[agent_index]
            state = self.policy_state(agent_index, pos)
        print(f"The best policy for agent {agent_index}: {policy}")
        return(policy)

    def _get_policy_noloop(self, agent_index):
        pos = self.starts[agent_index]
        move = [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
        done = False
        policy = []
        state =  self.policy_state(agent_index,pos)
        while not done:
            s = np.where(state.flatten() == agent_index+1)
            Q_prov = self.Q[agent_index]
            Q_prov[Q_prov == 0] = -100000
            best_action = np.argmax(Q_prov[s])
            policy.append(best_action)
            pos = tuple(np.add(pos, move[best_action]))
            done = pos == self.destinations[agent_index]
            state = self.policy_state(agent_index, pos)
        print(f"The best policy for agent {agent_index}: {policy}")
        return(policy)

    def render(self):
        self.ax.cla()  # Clear the axis to redraw
        grid = self._get_state_render()  # Get the grid state

        # Only fill spaces that are not empty
        filled = np.zeros(grid.shape, dtype=bool)
        filled[grid != 0] = True

        # Create a color mapping for the grid
        colors = np.empty(grid.shape, dtype=object)
        colors[grid == -1] = 'red'    # Obstacles in red
        colors[grid == 1] = 'green'   # Agent in green
        colors[grid == -2] = 'yellow' # Destination in yellow
        colors[grid == 2] = "blue"
        colors[grid == -3] = 'orange'

        # Plot the grid using voxels
        self.ax.voxels(filled, facecolors=colors, edgecolors='k')

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
    Further methods can be defined if necessary (e.g., random_state(), start_state(), animation, ...)
    """

m = MultiAgentThreeDGridEnv()

m.reset()
m.episode(epsilon= 0.8)
