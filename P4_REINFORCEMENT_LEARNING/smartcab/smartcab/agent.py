import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q    = {} # Q-learning table as a dictionary
        self.time = 0  # keeping current timestep information for calculating alpha and epsilon
        self.trial = 0 # trail number (1-100)

        # variables of previous state,action and reward info used to update Q-table
        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = None

        # learning paramters
        self.use_epsilon = False    # whether to use epsilon strategy?
        self.use_alpha   = False    # whether to keep learnt information using alpha learning rate
        self.gamma       = 0        # discounting factor

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = None
        self.trial = self.trial + 1
    def choose_action(self, state):
        """ choose which action to take for a particular state as per Q-learning """

        # if state is not present in Q-table, initialize correspnding state-action pairs values
        if state not in self.Q:
            self.Q[state] = {}
            for act in self.env.valid_actions:
                self.Q[state][act] = 0

        # epsilon-greedy strategy
        # if epsilon-probability select a random action, else choose action corresponding to max-value in Q-table
        self.epsilon = 1.0/self.time
        if self.use_epsilon and random.random() < self.epsilon:
            # exploration
            print "RANDOM ACTION SELECTED"
            action = random.choice(self.env.valid_actions)
        else:
            # exploitation
            maxQValue = max(self.Q[state].values())
            max_actions = [key for key, value in self.Q[state].iteritems() if value == maxQValue]
            action = random.choice(max_actions)
        return action

    def update_qvalue(self, state, action):
        """ update Q-value of previous state-action pair as per reward values as per current action selected """
        maxQValue = max(self.Q[state].values())
        if self.use_alpha:
            self.alpha = 1.0/self.time
        else:
            self.alpha = 1.0
        if self.prev_state != None:
            self.Q[self.prev_state][self.prev_action] = ((1-self.alpha)*self.Q[self.prev_state][self.prev_action]) \
                                                        + (self.alpha*(self.prev_reward + self.gamma *maxQValue))

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.time += 1

        # TODO: Update state
        state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        self.state = state
        action = self.choose_action(state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        #if reward < 0:
        #    self.env.id_last_neg = self.trial
        #if deadline == 0 and reward != 12:
        #    self.env.id_last_failure = self.trial

        # TODO: Learn policy based on state, action, reward
        self.update_qvalue(state, action)

        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
