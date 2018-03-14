from __future__ import division
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 800*10**6 # should be > 620 MPix
import os
from tqdm import tqdm

import fmow_helper

def extract_tile(row, output_size, context_ratio):
    assert row.full_path.endswith('.json')
    image_path = row.full_path[:-5]+'.jpg'
    img = Image.open(image_path)
    aspect = img.height / img.width
    if callable(context_ratio):
        physical_size = max(row.box2, row.box3/aspect) / img.width / aspect * row.width_m
        context_ratio = context_ratio(physical_size)
    extra = int(round((context_ratio-1) * max(row.box2, row.box3/aspect) / 2))
    img = img.crop((row.box0 - extra, row.box1 - int(round(extra*aspect)), row.box0+row.box2 + extra, row.box1+row.box3 + int(round(extra*aspect))))
    #scale = output_size / float(max(row.box2, row.box3))
    scale = output_size / max(img.width, img.height/aspect)
    img = img.resize([int(round(scale * v)) for v in (img.width, img.height/aspect)], Image.BICUBIC)
    pad_x = output_size - img.size[0]
    pad_y = output_size - img.size[1]
    img = img.crop((-(pad_x//2), -(pad_y//2), output_size-(pad_x//2), output_size-(pad_y//2)))
    img = np.asarray(img)
    assert img.shape == (output_size, output_size, 3)
    return img.tostring()

def star_extract_tile(args):
    return extract_tile(*args)

def dynamic_context(physical_size):
    t = 1 - (np.log2(physical_size) - 5) / (11-5)
    t = max(0, min(1, t))
    return 1.5 * 2**t

if __name__ == '__main__':
    import sys
    folds = sys.argv[1:]
    assert folds
    for fold in folds:
        for subset in ['complete', 'small']:
            for variable_size in (0,1):
                if fold=='test' and subset=='small':
                    continue

                output_size = 256
                context_ratio = dynamic_context if variable_size else 2
                ground = fmow_helper.csv_parse('working/metadata/boxes-%s-rgb.csv' % fold)

                if subset=='small':
                    ground = ground[ground.img_width==ground.groupby('ID').img_width.max().loc[ground.ID].values].drop_duplicates('ID')

                print("dataset has %d boxes and %d IDs" % (len(ground), len(ground.ID.value_counts())))
                print('%.2f Gpix input' % ground.eval('img_width*img_height/1e9').sum())
                print('%.2f Gpix output' % (len(ground) * output_size**2/1e9))

                root = '%s_%s_%d_varsize=%d' % (fold, subset, output_size, variable_size)
                os.makedirs('working/dataset', exist_ok=True)
                with open('working/dataset/tmp.%s_X.u8.tmp' % root, 'wb') as f:
                    from multiprocessing import Pool
                    pool = Pool(16)
                    tiles = pool.imap(star_extract_tile, ((row, output_size, context_ratio) for _,row in ground.iterrows()))
                    for tile in tqdm(tiles, desc="generate dataset %r" % root, unit="boxes", unit_scale=True, total=len(ground)):
                        f.write(tile)
                os.replace('working/dataset/tmp.%s_X.u8.tmp' % root, 'working/dataset/%s_X.u8' % root)

                if 'category' not in ground.columns:
                    ground['category'] = np.nan
                ground['category'].to_csv('working/dataset/%s_y.csv' % root)
                ground['ID'].to_csv('working/dataset/%s_group.csv' % root)
