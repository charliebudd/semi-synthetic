import numpy as np
from skimage.draw import line
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
import sknw


def compute_skeleton(mask):

    #####################################
    # Filling holes...

    mask = binary_fill_holes(mask)

    #####################################
    # Mask extrapolation...

    x, y = np.arange(mask.shape[1]), np.arange(mask.shape[0])
    x_grid, y_grid = np.meshgrid(x, y)

    points_x = x_grid[mask != 0]
    points_y = y_grid[mask != 0]

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

    def min_dist(p):
        x, y = p
        width, height = mask.shape[:2]
        dist_x = min(x, width - x)
        dist_y = min(y, height - y)
        return min(dist_x, dist_y)

    if min_dist(end) <  min_dist(start):
        start, end = end, start

    start = [start[1], start[0]]
    end = [end[1], end[0]]

    ######################
    
    vector = (np.array(start) - np.array(end)).astype(float)
    vector /= np.linalg.norm(vector)

    p = int(min(mask.shape) * 0.2)
    p2 = 2*p
    padded_mask = np.pad(mask, ((p2, p2), (p2, p2)))

    brush = np.copy(mask)
    brush[3:-3, 3:-3] = 0
    brush = np.pad(brush, ((p2, p2), (p2, p2)))

    for shift in range(p2):

        shift_vector = (vector * shift).astype(int)

        shifted_mask = np.roll(brush, shift_vector, axis=(1, 0))
        shifted_mask[p2:-p2, p2:-p2] = 0

        padded_mask = np.logical_or(padded_mask, shifted_mask)
    
    padded_mask = padded_mask[p:-p, p:-p].astype(float)
    padded_mask[p:-p, p:-p] += 0.1

    ##########################
    # skeleton

    for i in range(15):
        padded_mask = morphology.dilation(padded_mask)

    # padded_mask = binary_fill_holes(padded_mask)
    sk = morphology.skeletonize(padded_mask)
    skeleton = sk[p:-p, p:-p]

    #####################################
    # Constructing graph from skeleton...

    graph = sknw.build_sknw(skeleton)
    edges = graph.edges()
    nodes = graph.nodes()
        
    nodes = np.array([nodes[i]['o'] for i in nodes])

    for i in range(len(nodes)):
        nodes[i][0], nodes[i][1] = nodes[i][1], nodes[i][0]

    nodes = np.array(nodes).astype(np.int32).tolist()
    edges = np.array(edges).astype(np.int32).tolist()

    return {
        'nodes': nodes,
        'edges': edges,
        'tags': len(nodes) * ['visible', ]
    }


def crop_skeleton(skeleton, mask):

    edges = skeleton['edges']
    nodes = skeleton['nodes']
    tags = len(nodes) * ['visible', ]
    
    def get_connected_nodes(i):
        nodes = []
        edge_indices = []
        for j in range(len(edges)):
            edge = edges[j]
            if edge[0] == i:
                nodes.append(edge[1])
                edge_indices.append(j)
            elif edge[1] == i:
                nodes.append(edge[0])
                edge_indices.append(j)
        return nodes, edge_indices

    done = False

    while not done:

        done = True

        for home_node in range(len(nodes)):
            connected_nodes, edge_indices = get_connected_nodes(home_node)
            if len(connected_nodes) == 1:
                other_node = connected_nodes[0]

                start = nodes[home_node]
                end = nodes[other_node]
                
                line_points = list(zip(*line(*start[::-1], *end[::-1])))

                start_index = 0
                while start_index < len(line_points) and mask[line_points[start_index]] == 0:
                    start_index += 1

                if start_index == len(line_points):
                    print(edges)
                    print(nodes)

                    print(f"removing {home_node}, {edge_indices[0]}")
                    del(edges[edge_indices[0]])
                    del(nodes[home_node])
                    for edge in edges:
                        if edge[0] > home_node:
                            edge[0] -= 1
                        if edge[1] > home_node:
                            edge[1] -= 1
                    done = False
                    print(edges)
                    print(nodes)
                    break
                else:
                    nodes[home_node] = line_points[start_index][::-1]
            
    nodes = np.array(nodes).astype(np.int32).tolist()
    edges = np.array(edges).astype(np.int32).tolist()

    return {
        'nodes': nodes,
        'edges': edges,
        'tags': tags
    }

    # new_nodes = []
    # new_edges = []

    # for edge in skeleton['edges']:

    #     start_node, end_node = edge

    #     start = skeleton['nodes'][start_node]
    #     end = skeleton['nodes'][end_node]

    #     line_points = list(zip(*line(*start[::-1], *end[::-1])))

    #     start_index = 0
    #     end_index = len(line_points) - 1

    #     while start_index < end_index and mask[line_points[start_index]] == 0:
    #         start_index += 1
        
    #     while end_index > start_index and mask[line_points[end_index]] == 0:
    #         end_index -= 1

    #     start = (int(line_points[start_index][0]), int(line_points[start_index][1]))
    #     end = (int(line_points[end_index][0]), int(line_points[end_index][1]))

    #     if start_index != end_index:
    #         new_nodes.append(start[::-1])
    #         new_nodes.append(end[::-1])
    #         new_edges.append((start_node, end_node))
    
    # return {
    #     'nodes': new_nodes,
    #     'edges': new_edges,
    #     'tags': ['visible', 'visible']
    # }
