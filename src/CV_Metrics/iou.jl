# IoU implementation with Julia
# Ref: https://github.com/MartinThoma/algorithms/blob/master/CV/IoU/IoU.py

"""
    IoU(bb1, bb2)

Calculate the Intersection over Union (IoU) of two axis-aligned bounding boxes `bb1` and `bb2`.

Here, `bb1` and `bb2` are provided as `Dict` with keys = {"x1", "x2", "y1", "y2"}, where `x1`, `y1` are coordinates of top-left corner, and
`x2` and `y2` are coordinates of bottom-right corner.  
"""
function IoU(bb1, bb2)
    @assert bb1["x1"] < bb1["x2"]
    @assert bb1["y2"] < bb1["y1"]
    @assert bb2["x1"] < bb2["x2"]
    @assert bb2["y2"] < bb2["y1"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left || y_bottom < y_top
        return 0.0
    end
    
    # compute the intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both boxes
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y1"] - bb1["y2"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y1"] - bb2["y2"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / Float64(bb1_area + bb2_area - intersection_area)
    
    @assert iou >= 0.0
    @assert iou <= 1.0
    return iou
end
