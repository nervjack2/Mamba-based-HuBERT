import os 
import sys 
import numpy as np
from tqdm import tqdm

def cal_cross_prob(label_dir, phone_dir, n_label, n_phone):
    files_list = os.listdir(label_dir)
    prob_xy = np.zeros((n_label, n_phone), dtype=int)
    n_frame = 0
    for file_name in tqdm(files_list):
        label = np.load(os.path.join(label_dir, file_name))
        phone = np.load(os.path.join(phone_dir, file_name))
        l_len, p_len = label.shape[0], phone.shape[0]
        # The length of label and phone should only differ for a little
        assert abs(l_len - p_len) <= 8, f'{l_len} {p_len}'

        label, phone = label[:min(l_len, p_len)], phone[:min(l_len, p_len)]
        for l, p in zip(label, phone):
            if l >= n_label or p >= n_phone:
                continue 
            prob_xy[l,p] += 1 
            n_frame += 1 
    prob_xy = prob_xy/n_frame
    return prob_xy

def cal_purity(prob_xy, axis):
    max_ind = prob_xy.max(axis=axis)
    sum_ind = prob_xy.sum(axis=axis)
    cond_prob = max_ind/sum_ind
    purity = cond_prob.mean()
    return purity

def cal_purity_2(prob_xy, axis):
    max_ind = prob_xy.max(axis=axis)
    return max_ind.sum()

def main(
    label_dir, 
    phone_dir,
    output_file,
    n_label=100,
    n_phone=41
):
    print(f'Calculate purity with {n_label} cluster label and {n_phone} phone.')
    prob_xy = cal_cross_prob(label_dir, phone_dir, n_label, n_phone)
    phone_purity_2 = cal_purity_2(prob_xy, axis=1)
    cluster_purity_2 = cal_purity_2(prob_xy, axis=0)
    print(f'Got {phone_purity_2} phone purity.')
    print(f'Got {cluster_purity_2} cluster purity.')
    with open(output_file, 'a') as fp:
        fp.write(f'{phone_purity_2} {cluster_purity_2}\n')    

if __name__ == '__main__':
    label_dir, phone_dir = sys.argv[1], sys.argv[2]
    n_label, n_phone = int(sys.argv[3]), int(sys.argv[4])
    output_file = sys.argv[5]
    main(label_dir, phone_dir, output_file, n_label, n_phone)

