from viterbi import convert_to_list, viterbi
import numpy as np
from readfile import readfile
import copy

def new_emmi_prop(emit_p,obs_seq):

    not_p = {
        'Zero': {'demo': 0.0329175, 'video': 0.0029925, 'testimonial': 0.0155925, 'pricing': 0.1269675,
                 'blog': 0.2962575, 'payment': 0.0},
        'Aware': {'demo': 0.035343, 'video': 0.003213, 'testimonial': 0.056133, 'pricing': 0.136323, 'blog': 0.212058,
                  'payment': 0.0},
        'Considering': {'demo': 0.04788, 'video': 0.08208, 'testimonial': 0.01008, 'pricing': 0.12768, 'blog': 0.12768,
                        'payment': 0.0},
        'Experiencing': {'demo': 0.06384, 'video': 0.14364, 'testimonial': 0.00504, 'pricing': 0.04104, 'blog': 0.06384,
                         'payment': 0.0},
        'Ready': {'demo': 0.0039, 'video': 0.2223, 'testimonial': 0.0399, 'pricing': 0.018525, 'blog': 0.0494,
                  'payment': 0.0},
        'Lost': {'demo': 0.00729828, 'video': 0.00729828, 'testimonial': 0.02234628, 'pricing': 0.03802788,
                 'blog': 0.18063243, 'payment': 0.0},
        'Satisfied': {'demo': 0.0, 'video': 0.0, 'testimonial': 0.0, 'pricing': 0.0, 'blog': 0.0, 'payment': 0.16929}
    }

    p = copy.deepcopy(not_p)
    for k, data in not_p.items():
        for i in obs_seq:
            if i in data:
                continue
            else:
                if i == 'empty':
                    y = 1
                    for i, j in data.items():
                        y = y * (1 - j)
                    p[k]['empty'] = y
                else:
                    spliti = i.strip().split(',')
                    y = 1
                    for b, j in data.items():
                        if b in spliti:
                            y = y * j
                        else:
                            y = y * (1 - j)
                    p[k][i] = y
    return p

if __name__ == '__main__':


    start_p = {'Zero': 1, 'Aware': 0, 'Considering': 0, 'Experiencing': 0, 'Ready': 0, 'Lost': 0, 'Satisfied': 0}

    trans_p = {
        'Zero': {'Zero': 0.6, 'Aware': 0.4, 'Considering': 0.0, 'Experiencing': 0.0, 'Ready': 0.0, 'Lost': 0.0,'Satisfied': 0.0},
        'Aware': {'Zero': 0.0, 'Aware': 0.49, 'Considering': 0.3, 'Experiencing': 0.0, 'Ready': 0.01, 'Lost': 0.2,'Satisfied': 0.0},
        'Considering': {'Zero': 0.0, 'Aware': 0.0, 'Considering': 0.48, 'Experiencing': 0.2, 'Ready': 0.02,'Lost': 0.3, 'Satisfied': 0.0},
        'Experiencing': {'Zero': 0.0, 'Aware': 0.0, 'Considering': 0.0, 'Experiencing': 0.4, 'Ready': 0.3,
                         'Lost': 0.3, 'Satisfied': 0.0},
        'Ready': {'Zero': 0.0, 'Aware': 0.0, 'Considering': 0.0, 'Experiencing': 0.0, 'Ready': 0.8, 'Lost': 0.2,'Satisfied': 0.0
                  },
        'Lost': {'Zero': 0.0, 'Aware': 0.0, 'Considering': 0.0, 'Experiencing': 0.0, 'Ready': 0.0,'Lost': 1.0, 'Satisfied': 0.0
                 },
        'Satisfied': {'Zero': 0.0, 'Aware': 0.0, 'Considering': 0.0, 'Experiencing': 0.0, 'Ready': 0.0,'Lost': 0.0,'Satisfied': 1.0
                       }

    }

    emit_p = {
        'Zero': {'demo': 0.1, 'video': 0.01, 'testimonial': 0.05, 'pricing': 0.3, 'blog': 0.5, 'payment': 0.0},
        'Aware': {'demo': 0.1, 'video': 0.01, 'testimonial': 0.15, 'pricing': 0.3, 'blog': 0.4, 'payment': 0.0},
        'Considering': {'demo': 0.2, 'video': 0.3, 'testimonial': 0.05, 'pricing': 0.4, 'blog': 0.4, 'payment': 0.0},
        'Experiencing': {'demo': 0.4, 'video': 0.6, 'testimonial': 0.05, 'pricing': 0.3, 'blog': 0.4, 'payment': 0.0},
        'Ready': {'demo': 0.05, 'video': 0.75, 'testimonial': 0.35, 'pricing': 0.2, 'blog': 0.4, 'payment': 0.0},
        'Lost': {'demo': 0.01, 'video': 0.01, 'testimonial': 0.03, 'pricing': 0.05, 'blog': 0.2, 'payment': 0.0},
        'Satisfied': {'demo': 0.4, 'video': 0.4, 'testimonial': 0.01, 'pricing': 0.05, 'blog': 0.5, 'payment': 1.0},
    }

    id_to_states = { 0 : 'Zero', 1: 'Aware',2: 'Considering', 3: 'Experiencing',4: 'Ready',5: 'Lost',6: 'Satisfied'}
    obs_seq, obs_seq_id = readfile("hmm_customer_1586733275338.txt")
    new_emmi_prop = new_emmi_prop(emit_p, obs_seq)

    tran_prob = convert_to_list(trans_p)
    emmi_prob = convert_to_list(new_emmi_prop)
    init_prob = np.array(list(start_p.values()))
    hidden_s, max_prob = viterbi(tran_prob, emmi_prob, init_prob, obs_seq_id)

    opt_seq = []
    for data in hidden_s:
        opt_seq.append(id_to_states[data])

    print("Observation sequence : " + str(obs_seq))
    print()
    print("Transition Probability: "  + str(trans_p))
    print()
    print("Emmision Probability: "  + str(new_emmi_prop))
    print()
    print("Optimal state sequence: " + str(opt_seq))
    print()
    print("Maximum Probability: " + str(max_prob))

