
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import heapq # Default python script
import Sum_Tree as ST # Requires Sum_Tree script

def generate_transitionList():
    right_actions = []
    for n in range(N):
        right_actions.append(0 if n%2==0 else 1)

    sequences_list = []

    sequences_list.append([1])
    for state in range(N-1):
       current_sequence = [right_actions[i] for i in range(state+1)]
       current_sequence.append(right_actions[state])
       sequences_list.append(copy.copy(current_sequence))
    sequences_list.append(right_actions)

    random_sequence = np.arange(len(sequences_list))
    random.shuffle(random_sequence)
    for seq_num in random_sequence:
        current_sequence = sequences_list[seq_num]
        curr_state = 0
        for action in current_sequence:
            if action == right_actions[curr_state]:
                next_state = curr_state+1
            else:
                next_state = 0
            reward = 0
            if curr_state == N-1 and action == right_actions[N-1]:
                next_state = 0
                reward = 1
            transition_list.append([curr_state, action, reward, next_state])

            curr_state = next_state

def generate_groundTruth():
    print('Finding ground truth')
    updates = 0
    while updates<50000000:
        transition = random.sample(transition_list,1)[0]
        Q_values_true[transition[1],transition[0]] += ETA*(transition[2] + GAMMA* np.max(Q_values_true[:,transition[3]])-Q_values_true[transition[1],transition[0]])
        updates+=1

def generate_uniqueTransitionList():
    for transition in transition_list:
        if transition not in transition_list_unique:
            transition_list_unique.append(transition)

def find_transition():
    if agent == 'R':
        return None,random.sample(transition_list,1)[0]
    elif agent == 'O':
        mse_min = 10
        best_transition = None
        for transition in transition_list_unique:
            Q_values_test = copy.copy(Q_values)
            Q_values_test[transition[1], transition[0]] += ETA * (
                        transition[2] + GAMMA * np.max(Q_values_test[:, transition[3]]) - Q_values_test[transition[1], transition[0]])
            MSE = np.sum(abs(Q_values_test - Q_values_true) ** 2) / Q_values.size
            if MSE<mse_min:
                mse_min = MSE
                best_transition = transition
        return None,best_transition
    elif agent == 'TD':
        return None,transition_list[error_heap.return_maxTDTransition()]
    elif agent == 'SPTD':
        sum = np.random.uniform(0,priority_tree.tree_nodes[0],1)
        transition_index,_ = priority_tree.get_priority(sum,0)

        return transition_index,transition_list[transition_index]

class Heap:
    def __init__(self):
        self.TDError_heap = np.random.normal(-100, 1, len(transition_list)).tolist()
        self.TDError_transitions = copy.copy(np.array(self.TDError_heap))
        heapq.heapify(self.TDError_heap)
        self.current_transition = None

    def return_maxTDTransition(self):
        max_TDError = self.TDError_heap[0]
        self.current_transition = np.where(self.TDError_transitions == max_TDError)[0][0]
        return self.current_transition

    def update_heap(self,td_mag):
        self.TDError_transitions[self.current_transition] = -td_mag
        heapq.heapreplace(self.TDError_heap,-td_mag)

def run_episode():
    updates = 0
    while True:
        if updates in np.arange(0, max_updates[agent_no], max_updates[agent_no] / 10) and agent == 'O':
            print('Update No : {}'.format(updates))
        transition_index, transition = find_transition()
        td_error = ETA * (
                    transition[2] + GAMMA * np.max(Q_values[:, transition[3]]) - Q_values[transition[1], transition[0]])
        if agent == 'TD':
            error_heap.update_heap(abs(td_error))
        elif agent == 'SPTD':
            priority = abs(td_error) + 0.00001
            priority_tree.update_leaf(transition_index, priority)
        Q_values[transition[1], transition[0]] += td_error
        MSE[updates] += np.sum(abs(Q_values - Q_values_true) ** 2) / Q_values.size
        if MSE[updates] < threshold or updates > max_updates[agent_no] - 2:
            break
        updates += 1

N = 1024# number of states
ETA = 0.25 # Learning rate
GAMMA = 1- 1/N
threshold = 10**-3
transition_list = [] # List of all transitions
transition_list_unique = [] # Listing all unique transitions
Q_values_true = np.random.normal(0, 0.1, (2, N)) # Init true Q-table

generate_transitionList() # Generate list of all transitions
generate_groundTruth() # Obtain the ground truth Q-table
generate_uniqueTransitionList() # Generate unique transitions list

Agents = ['SPTD','TD','R'] # SPTD - Stochastic Proportional TD, TD - Greedy TD, 'R' - Random, 'O' - Oracle
MSE_All = [] # Stores Avg. MSE from all agents
max_updates = [5000000,5000000,5000000,5000000] # Can tune this to reduce run time for different state lengths
Episodes = [5, 5, 5, 1] # Number of episodes for each agent
for agent_no,agent in enumerate(Agents):
    MSE = np.zeros(5000000)
    for episode in range(Episodes[agent_no]):
        print('Running {} for agent {}'.format(episode, agent))
        Q_values = np.random.normal(0,0.01,(2, N))
        if agent == 'TD':
            error_heap = Heap()
        elif agent == 'SPTD':
            priority_tree = ST.Sum_Tree(len(transition_list))
            priority_tree.priorities = np.random.normal(100, 1, len(transition_list))
            priority_tree.update_allLeaves()

        run_episode()

    MSE_All.append(MSE / Episodes[agent_no])

# Ploting mean squared errors
plt.figure(1)
MSE_plot = plt.subplot()
for agent_no,agent in enumerate(Agents):
    MSE_plot.plot(MSE_All[agent_no], label = Agents[agent_no])

MSE_plot.set_ylabel('MSE average')
MSE_plot.set_xlabel('Updates')
MSE_plot.set_xlim((0,5000000))
MSE_plot.legend()
