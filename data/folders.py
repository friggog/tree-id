#! /usr/local/bin/python3

import os
import glob
from shutil import copyfile

# prefixes = []

# for path in glob.glob('images/*'):
#     prefix = path.split('/')[-1].split('_')[0]
#     if prefix not in prefixes:
#         prefixes.append(prefix)

# for i, p in enumerate(prefixes):
#     print(i, end='\r')
#     for path in glob.glob('images/' + p + '_*'):
#         new_path = 'all/' + p + '/'
#         if not os.path.exists(new_path):
#             os.makedirs(new_path)
#         new_path += path.split('/')[-1]
#         os.rename(path, new_path)

for path in sorted(glob.glob('leafsnap/images/all/*/*')):
    if '-' in path:
        new_path = path.replace('all', 'lab')
    else:
        new_path = path.replace('all', 'field')
    folder_path = '/'.join(new_path.split('/')[:4])
    print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    copyfile(path, new_path)
