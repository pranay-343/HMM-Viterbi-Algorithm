# Viterbi algorithm
import numpy as np

def convert_to_list(dict_data):
    list_data = []
    for data in dict_data:
        list_data.append(list(dict_data[data].values()))
    return np.asarray(list_data)

def viterbi(tran_prob, emmi_prob, init_prob,obs_seq):
    S = tran_prob.shape[0]  # number of states
    obs_len = len(obs_seq)  # length of observation sequence
    # initialize V and prev matrices
    prev = np.zeros([obs_len - 1, S], dtype=int)
    V = np.zeros([S, obs_len])

    V[:, 0] = init_prob* emmi_prob[:, obs_seq[0]]
    # compute V and prev in a nested loop
    for t in range(1, obs_len):
        for n in range(S):
            temp_product = np.float128(V[:,t-1]*tran_prob[:,n]*emmi_prob[n,obs_seq[t]])
            prev[t-1,n] = np.argmax(temp_product)
            V[n,t] = np.float128(np.amax(temp_product))

    State = np.zeros(obs_len)
    max_prob = max(V[:,-1])
    last_state = np.argmax(V[:,-1])

    State[0] = last_state
    backtrack_index = 1
    for i in range(obs_len - 2, -1, -1):
        State[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
    State = np.flip(State, axis=0)

    return  State.astype(int) ,max_prob
