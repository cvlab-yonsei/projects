import numpy as np
import sys


def Read(filename):
    # Sanity Check
    if filename == None:
        print('readFlowFile: empty filename')
        
    TAG_FLOAT = 202021.25     # Check for this when READING the file
    
    idx = str.find(filename, '.')
    name_length = len(filename)
    end_idx = name_length - 1
    
    # Sanity Check
    if idx == end_idx or idx == -1:
        error('readFlowFile: extension required in filename %s', filename)
        
    if filename[idx:name_length] != '.flo':
        error('readFlowFile: filename %s should have extension ''.flo''', filename)

    fid = open(filename, 'r', encoding='latin-1')
    if fid == None:
        error('readFlowFile: could not open %s', filename)
    
    tmp =  np.fromfile(fid, dtype=np.float32)
    fid.close()
    fid = open(filename, 'r', encoding='latin-1')
    tmp1 = np.fromfile(fid, dtype=np.int32)
    fid.close()
    tag = tmp[0]
    width = tmp1[1]
    height = tmp1[2]

    # Sanity check
    if (tag != TAG_FLOAT):
        error('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)', filename)
    
    if (width < 1 or width > 99999):
        error('readFlowFile(%s): illegal width %d', filename, width)

    if (height < 1 or height > 99999):
        error('readFlowFile(%s): illegal height %d', filename, height)

    nBands = 2

    img = tmp[3:len(tmp)]
    img = np.reshape(img, [height, width, nBands])
    
    return img, width, height

#def Write(img, filename):
    

def makeColorwheel():
    RY = 15;
    YG = 6;
    GC = 4;
    CB = 11;
    BM = 13;
    MR = 6;

    ncols = RY + YG + GC + CB + BM + MR;

    colorwheel = np.zeros([ncols, 3]); # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255;
    colorwheel[0:RY, 1] = np.floor([255*i/RY for i in range(RY)])
    col = col+RY

    #YG
    colorwheel[col:col+YG, 0] = [255-np.floor(255*i/YG) for i in range(YG)]
    colorwheel[col:col+YG, 1] = 255;
    col = col+YG;

    #GC
    colorwheel[col:col+GC, 1] = 255;
    colorwheel[col:col+GC, 2] = np.floor([255*i/GC for i in range(GC)])
    col = col+GC;

    #CB
    colorwheel[col:col+CB, 1] = [255-np.floor(255*i/CB) for i in range(CB)]
    colorwheel[col:col+CB, 2] = 255;
    col = col+CB;

    #BM
    colorwheel[col:col+BM, 2] = 255;
    colorwheel[col:col+BM, 0] = np.floor([255*i/BM for i in range(BM)])
    col = col+BM;

    #MR
    colorwheel[col:col+MR, 2] = [255-np.floor(255*i/MR) for i in range(MR)]
    colorwheel[col:col+MR, 0] = 255;
    
    return colorwheel
    
def computeColor(u,v):
    
    nanIdx = np.where(np.logical_or(np.isnan(u),np.isnan(v)));
    u[nanIdx] = 0;
    v[nanIdx] = 0;

    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.power(u,2)+np.power(v,2))

    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) /2 * (ncols-1) + 1  # -1~1 maped to 1~ncols
   
    k0 = np.floor(fk)                # 1, 2, ..., ncols

    k1 = k0+1
    a = np.where(k1==ncols+1)
    k1[a] = 1

    f = fk - k0
    img = np.zeros([u.shape[0], u.shape[1], colorwheel.shape[1]])
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = np.zeros([u.shape[0], u.shape[1]])
        col1 = np.zeros([u.shape[0], u.shape[1]])
        for j in range(u.shape[0]):
            for k in range(u.shape[1]):
                col0[j,k] = tmp[int(k0[j,k])-1]/255
                col1[j,k] = tmp[int(k1[j,k])-1]/255
        col = (1-f)*col0 + f*col1
        
        idx = np.where(rad <= 1)
        col[idx] = 1-rad[idx]*(1-col[idx])    # increase saturation with radius
        non_idx = np.where(rad > 1)
        col[non_idx] = col[non_idx]*0.75            # out of range
        
        img[:,:,i] = np.int8(np.floor(255*col))
    
    return img

def flow2color(flow):
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    eps = 1e-10

    [height, width, nBands] = flow.shape

    if nBands != 2:
        error('flowToColor: image must have two bands')
    
    u = np.zeros([height, width])
    v = np.zeros([height, width])
    u = flow[:,:,0]
    v = flow[:,:,1]
    

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    # fix unknown flow
    idxUnknown = np.where(np.logical_or((abs(u)> UNKNOWN_FLOW_THRESH),(abs(v)> UNKNOWN_FLOW_THRESH)))
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    if np.max(u)>-999:
        maxu = np.max(u)
    if np.min(u)<999:
        minu = np.min(u)
    if np.max(v)>-999:
        maxv = np.max(v)
    if np.min(v)<999:
        minv = np.min(v)

    rad = np.sqrt(np.power(u,2)+np.power(v,2))
    if np.max(rad)>-1:
        maxrad = np.max(rad)

    print('max flow: %f flow range: u = %f .. %f; v = %f .. %f' %(maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)

    # compute color

    img = computeColor(u, v)  
    
    # unknown flow
    idxUnknown = np.array(idxUnknown)
    IDX = np.tile(idxUnknown, [1, 1, 3])
    IDX = tuple(map(tuple,IDX))
    img[IDX] = 0
    
    for i in range(height):
        for j in range(width):
            for k in range(img.shape[2]):
                if img[i,j,k]<0:
                    img[i,j,k] = img[i,j,k] + 256
    img = img/255
    
    return img
    
