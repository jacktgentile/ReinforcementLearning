import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    # returns environment state variables needed for the algorithm
    def discretize_state(self, state):
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]
        # state vars
        adj_wall_x = 0
        adj_wall_y = 0
        food_dir_x = 0
        food_dir_y = 0
        adj_body_top = 0
        adj_body_bottom = 0
        adj_body_left = 0
        adj_body_right = 0

        # assign food_dir_y
        if snake_head_x == food_x:
            food_dir_x = 0
        elif snake_head_x > food_x:
            food_dir_x = 1
        else:
            food_dir_x = 2
        # assign food_dir_y
        if snake_head_y == food_y:
            food_dir_y = 0
        elif snake_head_y > food_y:
            food_dir_y = 1
        else:
            food_dir_y = 2
        # assign body state vars
        for body_segment in snake_body:
            # check top
            if adj_body_top == 0 and (body_segment[0] == snake_head_x and body_segment[1] == snake_head_y-1):
                adj_body_top = 1
                continue
            # check bottom
            if adj_body_bottom == 0 and (body_segment[0] == snake_head_x and body_segment[1] == snake_head_y+1):
                adj_body_bottom = 1
                continue
            # check left
            if adj_body_left == 0 and (body_segment[1] == snake_head_y and body_segment[0] == snake_head_x-1):
                adj_body_left = 1
                continue
            # check right
            if adj_body_right == 0 and (body_segment[1] == snake_head_y and body_segment[0] == snake_head_x+1):
                adj_body_right = 1
                continue
        # assign adj_wall_x
        if snake_head_x == utils.WALL_SIZE:
            adj_wall_x = 1
        elif snake_head_x == (utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE):
            adj_wall_x = 2
        # assign adj_wall_y
        if snake_head_y == utils.WALL_SIZE:
            adj_wall_y = 1
        elif snake_head_y == (utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE):
            adj_wall_y = 2
        # return values
        return (adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_body_top, adj_body_bottom, adj_body_left, adj_body_right)

    # find R(s) when training Q table
    def state_reward(self, state):
        # TODO return 1 when snake eats food, -1 when snake dies, -0.1 otherwise
        # is_dead determination taken from move in Snake

        # colliding with the snake body or going backwards while its body length
        # greater than 1
        if len(state[2]) >= 1:
            for seg in state[2]:
                if state[0] == seg[0] and state[1] == seg[1]:
                    return -1

        # moving towards body direction, not allowing snake to go backwards while
        # its body length is 1
        if len(state[2]) == 1:
            if state[2][0] == (state[0], state[1]):
                return -1

        # collide with the wall
        if (state[0] < utils.GRID_SIZE or state[1] < utils.GRID_SIZE or
            state[0] + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE or state[1] + utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE):
            return -1

        # check if eating food
        if state[0] == state[3] and state[1] == state[4]:
            return 1

        return -0.1

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        # calculate environment variables from state
        s_tuple = self.discretize_state(state)

        # copy arrays for easy reads
        N_values = self.N[s_tuple[0],s_tuple[1],s_tuple[2],s_tuple[3],s_tuple[4],s_tuple[5],s_tuple[6],s_tuple[7],:]
        Q_values = self.Q[s_tuple[0],s_tuple[1],s_tuple[2],s_tuple[3],s_tuple[4],s_tuple[5],s_tuple[6],s_tuple[7],:]
        a_start = random.randint(0,3)
        # determine a with max f value according to documentation
        best_action = self.actions[a_start]
        best_f = -1
        for a_offset in range(0,4):
            index = (a_offset + a_start) % 4
            cur_f = Q_values[index]
            if N_values[index] < self.Ne:
                cur_f = 1
            if cur_f > best_f:
                best_action = index
                best_f = cur_f
        # update N table
        self.N[s_tuple[0],s_tuple[1],s_tuple[2],s_tuple[3],s_tuple[4],s_tuple[5],s_tuple[6],s_tuple[7],best_action] += 1
        # if training, update Q table values. If not, just return action
        if not self.train:
            return best_action
        # calculate R(s)
        reward = self.state_reward(state)
        # determine s' (the next state after best_action is taken)
        if best_action == 0:
            # move up
            state[1] -= utils.GRID_SIZE
        elif best_action == 1:
            # move down
            state[1] += utils.GRID_SIZE
        elif best_action == 2:
            # move left
            state[0] -= utils.GRID_SIZE
        else:
            # move right
            state[0] += utils.GRID_SIZE
        # recalculate environment variables
        s_prime = self.discretize_state(state)
        max_Q = np.amax(self.Q[s_prime[0],s_prime[1],s_prime[2],s_prime[3],s_prime[4],s_prime[5],s_prime[6],s_prime[7],:])
        # calculate alpha (learning_rate)
        learning_rate = self.C / (self.C + N_values[best_action] + 1)
        # calculate delta value and update Q
        cur_Q = Q_values[best_action]
        delta = learning_rate * (reward + self.gamma * max_Q - cur_Q)
        self.Q[s_tuple[0],s_tuple[1],s_tuple[2],s_tuple[3],s_tuple[4],s_tuple[5],s_tuple[6],s_tuple[7],best_action] += delta
        return best_action
