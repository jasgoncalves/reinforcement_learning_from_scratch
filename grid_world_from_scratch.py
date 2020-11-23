# Based on the example of the Bellman Equation implementation of Grid World from
# https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff
# Reinforcement Learning — Implement Grid World
# Introduction of Value Iteration, May 4, 2019
# Jeremy Zhang

# Each action is determistic, not stochastic, which means if the agent intends
# to go somewhere, it will go there, vs having a probability of ending up there

import numpy as np

# global vars
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            print("winning state found, reward 1")
            return 1
        elif self.state == LOSE_STATE:
            print("Losing state found, reward -1")
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        ------------------
        0 | 1 | 2 | 3 |
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state is legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS - 1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS - 1)):
                    if nxtState != (1, 1):
                        return nxtState

            return self.state

        def showBoard(self):
            for i in range(0, BOARD_ROWS):
                print("------------------------")
                out = "| "
                for j in range(0, BOARD_COLS):
                    if self.board[i, j] == 1:
                        token = '*'
                    if self.board[i, j] == -1:
                        token = 'z'
                    if self.board[i, j] == 0:
                        token = '0'
                    out += token + " | "
                print(out)
            print("------------------------")


# Agent of the player
class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0

    def chooseAction(self):
        # choose the action with the most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        print("I have chosen an action to take: ", action)
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # loop until the end of the game, back propagate the rewards
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward value
                self.state_values[self.State.state] = reward  # this is aparently optional
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                print("Okay, so I'm at the starting position (2, 0)")
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, the agent then reaches the next state
                self.State = self.takeAction(action)
                # mark if it's the end
                self.State.isEndFunc()
                print("nxt.state", self.State.state)
                print("-----------------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print("-------------------------------------")
            out = "| "
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + " | "
            print(out)
        print("--------------------------------")


if __name__ == "__main__":
    ag = Agent()
    for count in range(10):
        print("--------STARTING-------------")
        print(ag.showValues())
        ag.play(5)
        print(ag.showValues())
        print("----------END----------------")
