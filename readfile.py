import numpy as np

def readfile(filename):
    file = open(filename, "r")
    obs_to_id = {'demo': 0, 'video': 1, 'testimonial': 2, 'pricing': 3, 'blog': 4, 'payment': 5}
    obs_seq = []
    obs_seq_id = []
    g = 5
    for line in file:
        if line.startswith("# States"):
            break
        if line.startswith("#"):
            continue
        if line == '\n':
            if 'empty' not in obs_to_id:
                g += 1
                obs_to_id['empty'] = g
                obs_seq.append('empty')
                obs_seq_id.append(g)
            else:
                obs_seq.append('empty')
                obs_seq_id.append(obs_to_id['empty'])
        else:
            k = line.strip().lower()
            if k not in obs_to_id:
                g += 1
                obs_to_id[k] = g
            obs_seq.append(k)
            obs_seq_id.append(obs_to_id[k])

    return np.asarray(obs_seq),np.asarray(obs_seq_id)