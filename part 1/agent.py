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

    def discretize_state(state):
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
        return adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_body_top, adj_body_bottom, adj_body_left, adj_body_right

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
        adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_body_top,
            adj_body_bottom, adj_body_left, adj_body_right = discretize_state(state)

        # copy arrays for easy reads
        N_values = self.N[adj_wall_x,adj_wall_y,food_dir_x,food_dir_y,adj_body_top,adj_body_bottom,adj_body_left,adj_body_right,:]
        Q_values = self.Q[adj_wall_x,adj_wall_y,food_dir_x,food_dir_y,adj_body_top,adj_body_bottom,adj_body_left,adj_body_right,:]
        # determine a with max f value according to documentation
        best_action = self.actions[0]
        best_f = -1
        for a in self.actions:
            cur_f = Q_values[a]
            if N_values[a] < self.Ne:
                cur_f = 1
            if cur_f > best_f:
                best_action = a
                best_f = cur_f
        # update N table
        self.N[adj_wall_x,adj_wall_y,food_dir_x,food_dir_y,adj_body_top,adj_body_bottom,adj_body_left,adj_body_right,best_action] += 1
        # if training, update Q table values
        if not self.train:
            return best_action
        learning_rate = self.C / (self.C + N_values[best_action] + 1)
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
        # recalculate environment vars
        adj_wall_x, adj_wall_y, food_dir_x, food_dir_y, adj_body_top,
            adj_body_bottom, adj_body_left, adj_body_right = discretize_state(state)
        # TODO calculate delta
        delta = 0

        return best_action
