import numpy as np
import pandas as pd
from tqdm import tqdm

import fmow_helper

BOXCOL = 'box0 box1 box2 box3'.split()

def fakegen(pbox):
    mask = np.ones(len(pbox), dtype=bool)
    nbox = pbox.copy()
    i0 = 100
    for i in tqdm(range(2*i0), desc="generate false_detection boxes"):
        if not np.any(mask):
            continue
        n = np.sum(mask)
        pbm = pbox[mask]
        for col in BOXCOL[:2]:
            nbox.loc[mask, col] = np.floor(np.random.rand(n) * pbm['img_width' if col in ('box0','box2') else 'img_height']).astype(int)
        mu = 0.45
        sd = 0.8
        sigmoid = lambda t: 1 / (1+np.exp(-t))
        rs = sigmoid(np.log(mu) + sd * np.random.randn(n))
        asd = 0.6
        ra = np.exp(asd * np.random.randn(n) / 2)
        nbox.loc[mask, 'box2'] = np.floor(rs*ra * pbm.img_width).astype(int)
        nbox.loc[mask, 'box3'] = np.floor(rs/ra * pbm.img_width).astype(int)
        mask2 = mask.copy() & False
        mask2 |= (nbox.box2 <= 0)
        mask2 |= (nbox.box3 <= 0)
        mask2 |= (nbox.box0+nbox.box2 >= nbox.img_width)
        mask2 |= (nbox.box1+nbox.box3 >= nbox.img_height)
        if i < i0:
            mask2 |= np.minimum(np.minimum(nbox.box0+nbox.box2, pbox.box0+pbox.box2) - np.maximum(nbox.box0, pbox.box0),
                                np.minimum(nbox.box1+nbox.box3, pbox.box1+pbox.box3) - np.maximum(nbox.box1, pbox.box1)) > 0
        mask = mask2
    return nbox.assign(category='false_detection')

def full_fakegen(btrain, seed):
    from scipy.stats import poisson
    obj_width_m = btrain.groupby('obj_id').width_m.first()
    obj_rate = pd.Series([1/6, 1/3, 5/6], [500, 1500, 5000]).loc[obj_width_m].values
    obj_num = pd.Series(poisson.rvs(obj_rate, random_state=seed), obj_width_m.index)
    obj_first_id = 9*10**5 + obj_num.shift().fillna(0).astype(int).cumsum()
    box_num = obj_num.loc[btrain.obj_id].values
    jumps = np.cumsum(box_num)
    i = np.cumsum(pd.value_counts(jumps).loc[np.arange(jumps[-1])].fillna(0).astype(int).values)
    pos = np.arange(len(i)) - np.r_[[0], jumps[:-1]][i]
    assert (pd.value_counts(i).loc[np.arange(len(box_num))].fillna(0).astype(int) == box_num).all()
    src = btrain.iloc[i]
    out = fakegen(src)
    out['ID'] = obj_first_id.loc[src.obj_id].values + pos
    out['visible'] = 1
    out['category'] = 'false_detection'
    out.index = 10**6 + np.arange(len(out))
    return out


def make_fake_boxes():
    btrain = fmow_helper.csv_parse('working/metadata/boxes-trainval-rgb.csv')
    bfake = full_fakegen(btrain.query('fold=="train"').iloc[:], seed=100)
    assert (bfake.obj_id.groupby(bfake.ID).max()==bfake.obj_id.groupby(bfake.ID).min()).all()
    assert bfake.groupby('ID').size().groupby(bfake.obj_id.groupby(bfake.ID).first()).std().max()==0
    fmow_helper.csv_trim(bfake).to_csv('working/metadata/boxes-fakev1-rgb.csv')

def make_train():
    ground = pd.concat([fmow_helper.csv_parse('working/metadata/boxes-%s-rgb.csv' % xfold)
                        for xfold in 'fakev1 trainval'.split()])
    fmow_helper.csv_trim(ground).to_csv('working/metadata/boxes-training-rgb.csv')

if __name__ == '__main__':
    make_fake_boxes()
    make_train()
