from __future__ import division

import numpy as np

def non_max_suppression_fast(boxes, overlapThresh, logger=None, pbar=None):
    #if boxes.dtype.kind == "i":
    #    boxes = boxes.astype("float")
    distances =boxes[:,4]
    boxes = boxes[:,0:4]
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (y1 - x1 + 1) * (y2 - x2 + 1)
    #area = (y1 - x1) * (y2 - x2)
    idxs = np.argsort(y2)

    if np.isinf(area).any() and logger:
        logger.log(
            '\tNMS: some are infinite',
            pbar=pbar
        )
    if np.isnan(area).any() and logger:
        logger.log(
            '\tNMS: some are Nan',
            pbar=pbar
        )
    if (not len(area.nonzero()[0]) == len(area)) and logger:
        logger.log(
            '\tNMS: some are zero',
            pbar=pbar
        )
    remaining_elements = list(range(boxes.shape[0]))
    while len(remaining_elements) > 0  :
        last = remaining_elements.pop(0)
        indexes_to_remove =[]
        test = True
        x1last,y1last,x2last, y2last = x1[last], y1[last] , x2[last] , y2[last]
        if x1last > x2last :
            x1last, y1last,  x2last, y2last = x2last, y2last, x1last, y1last
        for i in range(len( remaining_elements )) :
            el = remaining_elements[i]
            x11, y11, x22, y22 = x1[el] , y1[el] , x2[el], y2[el]
            if x11 > x22 :
                x11,y11,x22,y22 = x22, y22,x11,y11
            xx1 = max(x1last, x11)
            yy1 = min(y1last, y11)
            xx2 = max(x2last, x22)
            yy2 = min(y2last, y22)
            #w = max(0, yy1-xx1)
            #h = max(0, yy2-xx2)
            w = max(0, yy1-xx1+1)
            h = max(0, yy2-xx2+1)
            overlap= (w*h) / area[el]
            if overlap > overlapThresh :
                if distances[last] < distances[el] :
                    test = False
                else :
                    indexes_to_remove.append(i)
        if test :
            pick.append(last)
        popped = 0
        for j in indexes_to_remove :
            remaining_elements.pop(j-popped)
            popped +=1
        """
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        print(" last point  :  {} {} {} {} ".format(x1[i] , y1[i] , x2[i]  , y2[i] ))
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        print("xx1")
        print(xx1)

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        print("overlap")
        print(overlap)
        idxs = np.delete(
            idxs,
            np.concatenate((
                [last],
                np.where(overlap > overlapThresh)[0]
            ))
        )
        """
    return boxes[pick], pick

