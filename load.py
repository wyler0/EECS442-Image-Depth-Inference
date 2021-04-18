import matplotlib.pyplot as plt
from diode import DIODE, plot_depth_map

meta_fname = './diode_meta.json'
data_root = './dataset'

if __name__ == '__main__':

    for split in ('train', 'val', 'test'):
        for scene_type in ('outdoor', 'indoors'):
            dset = DIODE(
                meta_fname, data_root, splits=split, scene_types=scene_type
            )
            scenes = dset.meta[split][scene_type]
            num_scenes = len(scenes)
            num_scans = sum([ len(v) for k, v in scenes.items() ])
            print('{:8} {:8} {:<2} scenes {} scans {:>5} images'.format(
                split, scene_type, num_scenes, num_scans, len(dset))
            )
