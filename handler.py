import rrcf

def find_anomalies(input):
    # Set tree parameters
    num_trees = 40
    shingle_size = 1
    tree_size = 256

    # Create a forest of empty trees
    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree()
        forest.append(tree)

    inputPoints = list(map(lambda x: x['value'], input))

    points = rrcf.shingle(inputPoints, size=shingle_size)

    avg_codisp = {}
    disp = {}

    # For each shingle...
    for index, point in enumerate(inputPoints):
        # For each tree in the forest...
        for tree in forest:
            # If tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > tree_size:
                tree.forget_point(index - tree_size)
            # Insert the new point into the tree
            tree.insert_point(point, index=index)
            # Compute codisp on the new point and take the average among all trees
            if not index in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += tree.codisp(index) / num_trees
            disp[index] = tree.disp(index)

    output = []

    for i in range(len(input)):
        codisp = avg_codisp[i]

        point = {}
        point['value'] = input[i]['value']
        point['timestamp'] = input[i]['timestamp']
        point['isAnomaly'] = codisp > 40
        point['codisp'] = codisp
        output.append(point)

    return output


def detect(input, context):
    return find_anomalies(input)

def batch_detect(input, context):
    return list(map(lambda x: find_anomalies(x), input))
