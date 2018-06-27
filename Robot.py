import numpy as np
import operator

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.actions = {'u':0, 'r':0, 'd':0, 'l':0}
        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # No random choice when testing
            self.epsilon = self.epsilon0
        else:
            # Update parameters when learning
            self.t += 1
            self.epsilon = self.epsilon0 - 0.02 * self.t
        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # Return whether do random choice
            return np.random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # Return random choose aciton
                return np.random.choice(list(self.actions.keys()))
            else:
                # Return action with highest q value
                action = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
                return action
        elif self.testing:
            # choose action with highest q value
            action = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
            return action
        else:
            # Return random choose aciton
            return np.random.choice(list(self.actions.keys()))

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        q_predict = self.Qtable[self.state][action]
        q_target = r + self.gamma * max(list(self.Qtable[next_state].values()))
        self.Qtable[self.state][action] += self.alpha * (q_target - q_predict)  # 更新对应的 state-action 值


    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward

