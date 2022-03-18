import math

class td_qlearning:

  alpha = 0.2
  gamma = 0.9
  squares = ['W', 'X', 'Y', 'Z']
  actions = ['N', 'L', 'R', 'U', 'D']
  sub_table = [['W', 'N'], ['W', 'R'], ['W', 'D'], ['X', 'N'], ['X', 'L'], ['X', 'D'], ['Y', 'N'], ['Y', 'R'], ['Y', 'U'], ['Z', 'N'], ['Z', 'L'], ['Z', 'U']]
  q_table = []
  for i in squares:
    for x in sub_table:
      q_table.append([x[0] + i, x[1], 0])
  inputMap = []

  def maxNext(self, state):
    costArr = []
    for i in self.q_table:
      if state == i[0]:
        costArr.append(i[2])
    return max(costArr)

  def __init__(self, trial_filepath):
    # trial_filepath is the path to a file containing a trial through state space
    # Return nothing
    with open(trial_filepath) as rawInput:
      for i in rawInput:
        l = i.strip()
        self.inputMap.append(l.split(','))
    prev_node = None
    for i in self.inputMap:
      if prev_node != None:
        for x in self.q_table:
          if x[0] == prev_node[0] and x[1] == prev_node[1]:
            if prev_node[0][0] == prev_node[0][1]:
              reward = -1
            else:
              reward = 1
            x[2] = x[2] + self.alpha * \
                (reward + self.gamma *
                 self.maxNext(i[0]) - x[2])
      prev_node = i
    return None

# Q(s,a) <- Q(s,a) + alpha(r(s) + gamma max Q(s1, a1) - Q(s,a))
  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action
    for i in self.q_table:
      if state == i[0] and action == i[1]:
        return i[2]
    # Return the q-value for the state-action pair
    #return q
    return 0

  def policy(self, state):
    # state is a string representation of a state
    action = ''
    cost = -math.inf
    for i in self.q_table:
      if i[0] == state and i[2]>cost:
        action = i[1]
        cost = i[2]
    # Return the optimal action under the learned policy
    #return a
    return action