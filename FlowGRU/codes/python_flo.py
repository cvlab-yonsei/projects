import numpy as np

def readFLO(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return
        
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            #print('Reading %d x %d flo file' % (h, w))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            return data2D