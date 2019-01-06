from chainer.dataset import DatasetMixin
from pathlib import Path
import mhd
import numpy as np

class Dataset(DatasetMixin):
    __default_pattern = {
        'labels': '{patient_id}/*_crop_label_noskin.mhd',
        'projections': '{patient_id}/*_projections_bone_AP.mhd',
    }

    def __init__(self, root_dir, patients, patterns=None, with_spacing=False):
        if patterns is None:
            patterns = self.__default_pattern

        self.root_dir = root_dir
        self.files = {}
        self.with_spacing = with_spacing
        num_files = None
        for key, pattern in patterns.items():
            for patient_id in patients:
                self.files.setdefault(key, []).extend(Path(root_dir).glob(
                    pattern.format(patient_id=patient_id)
                ))

            if num_files is None:
                num_files = len(self.files[key])
                assert num_files != 0, 'File not found: ' + pattern
            else:
                assert num_files == len(self.files[key]), 'File not found: ' + pattern

    def __len__(self):
        return len(self.files['projections'])

    def get_example(self, index):
        labels, label_header = mhd.read(self.files['labels'][index])
        projections, proj_header = mhd.read(self.files['projections'][index])

        if labels.dtype.kind != 'i':
            labels = labels.astype(np.int32)

        if self.with_spacing:
            return (
                projections, labels,
                label_header['ElementSpacing'], proj_header['ElementSpacing']
            )
        else:
            return projections, labels
