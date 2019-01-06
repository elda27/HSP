from utils import mhd
import numpy as np
import os
import cv2

import datetime
import subprocess
import itertools

from utils.log_util import get_new_training_log_dir, get_training_log_dir

def bbox_ND(image):
    import itertools
    N = image.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(image, axis=ax)
        index = np.where(nonzero)[0]
        out.insert(0, index[0])
        out.insert(1, index[-1])
    return tuple(out)

def get_vcs_timestamp():

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    try: # for developer
        cmd = 'git log -n1 --format="%h"'
        vcs_version = subprocess.check_output(cmd, shell=True).strip()
        out = 'VCS-%s_TIME-%s' % (vcs_version, timestamp)
    except:
        out = 'TIME-%s' % timestamp

    return out

def load_image(filename):

    _, ext = os.path.splitext( os.path.basename(filename) )

    if ext in ('.mha', '.mhd'):
        [img, img_header] = mhd.read(filename)
        spacing = img_header['ElementSpacing']
        img.flags.writeable = True
        if img.ndim == 3:
            img = np.transpose(img, (1,2,0))

    elif ext in ('.png', '.jpg', '.bmp'):
        img = cv2.imread(filename)
        spacing = None

    else:
        raise NotImplementedError()

    return img, spacing

def save_image(filename, image, spacing=None):

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    _, ext = os.path.splitext( os.path.basename(filename) )

    if ext in ('.mha', '.mhd'):
        header = {}
        if spacing is not None:
            header['ElementSpacing'] = spacing
        if image.ndim == 2:
            header['TransformMatrix'] = '1 0 0 1'
            header['Offset'] = '0 0'
            header['CenterOfRotation'] = '0 0'
        elif image.ndim == 3:
            image = image.transpose((2,0,1))
            header['TransformMatrix'] = '1 0 0 0 1 0 0 0 1'
            header['Offset'] = '0 0 0'
            header['CenterOfRotation'] = '0 0 0'
        else:
            raise NotImplementedError()
        mhd.write(filename, image, header)

    elif ext in ('.png', '.jpg', '.bmp'):
        cv2.imwrite(filename, image)

    else:
        raise NotImplementedError()
