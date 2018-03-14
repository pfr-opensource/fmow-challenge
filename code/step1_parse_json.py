import pandas as pd
import json
from subprocess import check_output
import os
from tqdm import tqdm

def parse_jsons(fmow_rgb_root, dataset_name):
    subset,suffix = dataset_name.split('-')
    folders = { 'test': './', 'trainval': '{train/,val/}' }[subset]
    paths = sorted(check_output(["bash", "-c", "cd \"$1\"; find %s -iname '*_%s.json'" % (folders, suffix), 'find', fmow_rgb_root]).decode('ascii').splitlines())

    df_box = []

    for relative_path in tqdm(paths, desc="parse JSON for %s" % dataset_name, unit="images", unit_scale=True):
        path = os.path.join(fmow_rgb_root, relative_path)
        assert '\\' not in path
        if dataset_name.startswith('test'):
            fold = 'test'
        elif dataset_name.startswith('trainval'):
            fold = relative_path.split('/')[0]
            assert fold in ('train', 'val')
        if 'val/airport/airport_61/airport_61_3_' in path:
            print("info: skipping the file where MSRGB data is or was missing; this is normal")
            continue
        with open(path) as f:
            m = json.load(f)
            all_bb = m.pop('bounding_boxes')
            if isinstance(all_bb, dict):
                all_bb = [all_bb]
            m['wavelength_code'] = 100
            m['fold'] = fold
            m['full_path'] = path

            for i,bb in enumerate(sorted(all_bb, key=lambda bb: bb['ID'])):
                bb = dict(bb)
                box = bb.pop('box')
                if 'test-' in dataset_name:
                    assert 'visible' not in bb
                else:
                    bool_visible = { 'True': True, 'False': False }[bb['visible']]
                    bb['visible'] = int(bool_visible)
                bb['box_num'] = i
                for j,v in enumerate(box):
                    m['box%d'%(j)] = v
                for k,v in bb.items():
                    m[k] = v
                df_box.append(dict(m))

    os.makedirs('working/metadata', exist_ok=True)
    pd.DataFrame(df_box).to_csv('working/metadata/boxes-%s.csv' % dataset_name)

DATASETS = [
        'trainval-rgb',
        'test-rgb',
    ]

if __name__ == '__main__':
    import sys
    dataset_name, fmow_rgb_root = sys.argv[1:]
    parse_jsons(fmow_rgb_root, dataset_name)
