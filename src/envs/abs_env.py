import gym


class AbstractEnv(gym.Env):
    ''' Abstract class for defining an environment '''

    def __init__(self):

        '''
        Specifies information about the environment.
        @:param self.lows: a list (len = number of state features) of lower bounds for state features
        @:param self.highs: a list (len = number of state features) of higher bounds for state features
        @:param self.observation_space: observation space for Gym env from gym.spaces
        @:param self.action_space: action space for Gym env from gym.spaces
        '''
        self.lows = None
        self.highs = None
        self.observation_space = None
        self.action_space = None
        self.state_dim = 0

        self.max_penalty = -3

    def step(self, action):
        return None

    def close(self):
        pass

    def render(self):
        pass

    def reset(self):
        return None

    def render_state(self, x):
        ''' Renders single state x '''
        pass

    def realistic(self, x):
        ''' Returns a boolean indicating if x is a valid state in the environment (e.g. chess state without kings is not valid)'''
        return True

    def actionable(self, x, fact):
        ''' Returns a boolean indicating if all immutable features remain unchanged between x and fact states'''
        return True

    def get_actions(self, x):
        ''' Returns a list of actions available in state x'''
        return []

    def set_state(self, x):
        ''' Changes the environment's current state to x '''
        pass

    def check_done(self, x):
        ''' Returns a boolean indicating if x is a terminal state in the environment'''
        return False

    def equal_states(self, x1, x2):
        ''' Returns a boolean indicating if x1 and x2 are the same state'''
        return False

    def writable_state(self, x):
        ''' Returns a string with all state information to be used for writing results'''
        return None