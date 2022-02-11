import numpy as np

def filter_overlap(dets, overlapThresh):
    selected_dets_ids = []
    if dets:
        x1,y1,x2,y2, areas = convert_to_numpy(dets)
        undecided_dets_ids = np.argsort(y2)

        while len(undecided_dets_ids) > 0:
            newly_selected_id = undecided_dets_ids[-1]
            undecided_dets_ids = undecided_dets_ids[:-1]
            selected_dets_ids.append(newly_selected_id)
            overlap_areas = get_overlap_areas(x1,y1,x2,y2, undecided_dets_ids, newly_selected_id)
 
            base_areas = np.minimum(areas[newly_selected_id], areas[undecided_dets_ids])
            overlap_ratios = overlap_areas / base_areas
            undecided_dets_ids = undecided_dets_ids[overlap_ratios < overlapThresh]
 
    return [dets[j] for j in selected_dets_ids]

def convert_to_numpy(dets):
    boxes = [d['xyxy'] for d in dets]
    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1) * (y2 - y1)
    #area[area<1] = 1
    return x1,y1,x2,y2, area

def get_overlap_areas(x1,y1,x2,y2,undecided_dets_ids, newly_selected_id):
    xx1 = np.maximum(x1[newly_selected_id], x1[undecided_dets_ids])
    yy1 = np.maximum(y1[newly_selected_id], y1[undecided_dets_ids])
    xx2 = np.minimum(x2[newly_selected_id], x2[undecided_dets_ids])
    yy2 = np.minimum(y2[newly_selected_id], y2[undecided_dets_ids])
 
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap_areas = w*h
    return overlap_areas
 
def non_max_suppression_fast(dets, overlapThresh):
    #if overlapThresh == -1: # not using nms
    #    return [(classes[i], scores[i], tuple(boxes.astype('int')[i])) for i in range(len(classes))]
    #else:
    #    # if there are no boxes, return an empty list
    #    if len(boxes) == 0:
    #        return []
    ## if the bounding boxes integers, convert them to floats --
    #    # this is important since we'll be doing a bunch of divisions
    #    if boxes.dtype.kind == "i":
    #        boxes = boxes.astype("float")
    if not dets:
        return []
    else:
        boxes = np.array(dets, dtype=np.float32)

        # initialize the list of picked indexes 
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        #idxs = np.argsort(scores)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            areaMin = np.minimum(area[i], area[idxs[:last]])
            overlap = (w * h) / areaMin

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        #return boxes[pick].astype("int")

        ## format nms result into yolo format
        #dets_out = []
        #for i in pick:
        #    dets_out.append((classes[i], scores[i], tuple(boxes.astype('int')[i])))
        #return dets_out
        return [dets[j] for j in pick]


if __name__ == '__main__':
    dets = [['a', .5, [0,0,100,100]],['b', .8, [50,50,150,150]],['c', .7, [100,100,200,200]]]
    #print non_max_suppression_fast(dets, .2)
