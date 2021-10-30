import numpy as np
import cv2 as cv
from skimage.draw import line

def compute_skeleton(mask):

    print('pre blend:', mask.shape)

    num_labels, labels_im = cv.connectedComponents(mask)
    max_label = 1
    max_pixels = 0
    
    for i in range(1, num_labels):
        component = np.where(labels_im == i, 1, 0).astype(np.uint8)
        pixels = np.sum(component)
        if pixels > max_pixels:
            max_pixels = pixels
            max_label = i
    
    mask = np.where(labels_im == max_label, 1, 0)

    x, y = np.arange(mask.shape[1]), np.arange(mask.shape[0])
    x_grid, y_grid = np.meshgrid(x, y)

    points_x = x_grid[mask != 0][::10]
    points_y = y_grid[mask != 0][::10]

    m, c = np.polyfit(points_x, points_y, 1)

    vector = np.array([1, m])
    vector /= np.linalg.norm(vector)

    origin = [int(mask.shape[1] / 2), c + m * int(mask.shape[1] / 2)]

    min_dot = 1e9
    max_dot = -1e9

    for p_x, p_y in zip(points_x, points_y):
        local = np.array([p_x, p_y]) - origin
        dot = np.dot(local, vector)
        min_dot = min(min_dot, dot)
        max_dot = max(max_dot, dot)

    start = origin + vector * min_dot
    end = origin + vector * max_dot

    start = (int(start[1]), int(start[0]))
    end = (int(end[1]), int(end[0]))

    def clamp(p):
        x = min(max(p[0], 0), mask.shape[0] - 1)
        y = min(max(p[1], 0), mask.shape[1] - 1)
        return x, y

    start = clamp(start)
    end = clamp(end)
    
    print('pre blend:', start, end)

    def min_dist(p):
        x, y = p
        width, height = mask.shape[:2]
        dist_x = min(x, width - x)
        dist_y = min(y, height - y)
        return min(dist_x, dist_y)

    if min_dist(end) <  min_dist(start):
        start, end = end, start

    return {
        'nodes': [start, end],
        'edges': [(0, 1)]  
    }

def crop_skeleton(skeleton, mask):

    print('post blend:', mask.shape)

    new_nodes = []
    new_edges = []

    for edge in skeleton['edges']:

        start_node, end_node = edge

        start = skeleton['nodes'][start_node]
        end = skeleton['nodes'][end_node]

        line_points = list(zip(*line(*start, *end)))

        print(mask.shape)
        print(start, end)

        start_index = 0
        end_index = len(line_points) - 1

        while start_index < end_index and mask[line_points[start_index]] != 0:
            start_index += 1
        
        while end_index > start_index and mask[line_points[end_index]] != 0:
            end_index -= 1

        if start_index != end_index:
            new_nodes.append(start)
            new_nodes.append(end)
            new_edges.append((start_node, end_node))
    
    return {
        'nodes': new_nodes,
        'edges': new_edges  
    }
