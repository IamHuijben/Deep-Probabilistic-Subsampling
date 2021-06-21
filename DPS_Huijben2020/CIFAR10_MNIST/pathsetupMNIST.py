import os
in_dir = os.path.join(os.path.dirname(__file__), 'MNISTdata')
if not os.path.isdir(in_dir):
    os.makedirs(in_dir)

