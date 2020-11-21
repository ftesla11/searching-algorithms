import networkx as nx
import matplotlib.pyplot as plt
import csv
import json
import numpy as np

# Load the London tube graph from a file
def load_graph_from_csv(filename):
  with open(filename, 'r') as graph:
    Graphtype=nx.Graph()
    G = nx.parse_edgelist(graph, delimiter=', ', create_using=Graphtype,
                      nodetype=str, data=(('tube_line', str), ('avg_time', float), ('main', str), ('secondary', str)))
    return nx.Graph(G)

# Load labyrinth from json file
def load_graph_from_json(filename):
  with open(filename) as labyrinth_file:
    dict_labyrinth = json.load(labyrinth_file)
    return nx.Graph(dict_labyrinth)


# Display the weighted graph of the London tube map
def show_weighted_graph(networkx_graph, node_size, font_size, fig_size):
  # Allocate the given fig_size in order to have space for each node
  plt.figure(num=None, figsize=fig_size, dpi=80)
  plt.axis('off')
  # Compute the position of each vertex in order to display it nicely
  nodes_position = nx.spring_layout(networkx_graph) 
  # You can change the different layouts depending on your graph
  # Extract the weights corresponding to each edge in the graph
  edges_weights  = nx.get_edge_attributes(networkx_graph,'avg_time')
  # Draw the nodes (you can change the color)
  nx.draw_networkx_nodes(networkx_graph, nodes_position, node_size=node_size,  
                         node_color = ["orange"]*networkx_graph.number_of_nodes())
  # Draw only the edges
  nx.draw_networkx_edges(networkx_graph, nodes_position, 
                         edgelist=list(networkx_graph.edges), width=2)
  # Add the weights
  nx.draw_networkx_edge_labels(networkx_graph, nodes_position, 
                               edge_labels = edges_weights)
  # Add the labels of the nodes
  nx.draw_networkx_labels(networkx_graph, nodes_position, font_size=font_size, 
                          font_family='sans-serif')
  plt.axis('off')
  plt.show()


# Depth First Search
def dfs(graph, initial, goal, order):
    """
        :param graph   : networkx Graph structure
        :param initial : starting station
        :param goal    : destination
        :param order   : 0 or 1 determining which neighbor DFS should start with
    """

    # Nodes to be explored
    queue = [initial]

    # Nodes already visited
    visited = []

    # Parent of each node
    parent = {}

    while True:
        if not queue:
            return False
        node = queue.pop(0)

        # Keep track of visited nodes
        visited.append(node)

        if node == goal:
            path = []

            # Traceback the parents of the path found
            while parent[node] != initial:
                path.insert(0, parent[node])
                node = parent[node]

            # Insert start and end state to the path   
            path.insert(0, initial)
            path.append(goal)
            print('DFS Nodes visited: ', len(visited))
            print('DFS Solution cost: ', len(path)-1)
            return path
        
        neighbors = list(graph.neighbors(node))
        
        # Change the order of neighbors that DFS starts with
        if order != 0:
            neighbors = reversed(neighbors)

        # Expand the current node and explore the neighbors
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            # DFS Queuing strategy
            queue.insert(0, neighbor)
            # Insert parent node for trace back when goal found
            parent[neighbor] = node


# Breadth First Search
def bfs(graph, initial, goal, order):
    """
        :param graph   : networkx Graph structure
        :param initial : starting station
        :param goal    : destination station
        :param order   : 0 or 1 determining which neighbor BFS should start with
    """

    queue = [initial]
    visited = []
    node_map = {}

    while True:
        if not queue:
            return False
        node = queue.pop(0)

        # Keep track of visited nodes
        visited.append(node)

        if node == goal:
            path = []

            while node_map[node] != initial:

                # Insert the parent of current node, starting with exit
                path.insert(0, node_map[node])
                node = node_map[node]

            path.insert(0, initial)
            path.append(goal)
            print('BFS Nodes visited: ', len(visited))
            print('BFS Solution cost: ', len(path)-1)
            return path

        neighbors = list(graph.neighbors(node))

        if order == 1:
            neighbors = reversed(neighbors)

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            # BFS Queueing strategy
            queue.append(neighbor)
            # Insert parent node for path trace back when goal is found
            node_map[neighbor] = node



# Uniformed Cost Search which considers the time taken for transitions
def ucs(graph, initial, goal, heuristic):
    """
        :param graph        : networkx Graph structure
        :param initial      : starting station
        :param goal         : destination station
        :param heuristic    : defines the cost on which we determine the path
    """

    # Keep track of visited nodes
    visited = {}
    visited[initial] = (0, [initial], [])

    # Keep track of unexplored nodes
    unexplored = {}
    unexplored[initial] = (0, [initial], [])

    # Assuming transition duration is 2 minutes
    transition_cost = 2

    while unexplored:

        ordered_neighbors = sorted(unexplored, key=unexplored.get)
        node = ordered_neighbors[0]
        current_cost, path, current_tubelines = unexplored[node][0], unexplored[node][1], unexplored[node][2]

        # Remove the node from unexplored list
        unexplored.pop(node)

        neighbors = list(graph.neighbors(node))

        for neighbor in neighbors:
            weight = graph.get_edge_data(node, neighbor)[heuristic]
            total_cost = current_cost + weight

            tubeline = graph.get_edge_data(node, neighbor)['tube_line']

            # Add 2 minutes to the cost for every transition
            if current_tubelines and tubeline not in current_tubelines:
                total_cost += 2
            
            if (neighbor not in visited) or (visited[neighbor][0] > total_cost):

                new_tubelines = current_tubelines

                # Add a new transition to the path if tubeline isn't in the previous path
                if tubeline not in current_tubelines:
                    new_tubelines = current_tubelines+[tubeline]
  
                # Update the visited node and add it to unexplored for further expansion
                best_node = (total_cost, path+[neighbor], new_tubelines)
                visited[neighbor] = best_node
                unexplored[neighbor] = best_node

    print('Nodes visited: ', len(visited))
    print('Nodes to reach solution: ', len(visited[goal][1])-1)
    print('Number of transitions between tube lines: ', len(visited[goal][2])-1)
    return visited[goal]



# Compute Manhattan
def get_manhattan(node, goal):
    manhattan = np.abs(int(node[1]) - int(goal[1])) + np.abs(int(node[3]) - int(goal[3]))
    return manhattan

# Compute the distance between two nodes
def get_distance(node1, node2):
    distance = np.abs(int(node1[1]) - int(node2[1])) + np.abs(int(node1[3]) - int(node2[3]))
    return distance


# A* Algorithm using manhattan distance to solve labyrinth problem
def a_star(graph, initial, goal):
    """
        :param graph        : networkx Graph structure
        :param initial      : starting coordinates
        :param goal         : destination coordinates
    """
    queue = [initial]
    visited = []
    parent_dict = {}

    while True:
        if not queue:
            return False
        node = queue.pop(0)

        # Keep track of visited nodes
        visited.append(node)

        if node == goal:
            path = []
            total_weight = {}

            while parent_dict[node] != initial:

                # Insert the parent of current node, starting with exit
                path.insert(0, parent_dict[node])
                node = parent_dict[node]

            # Insert the initial state
            path.insert(0, initial)
            # Insert the final state
            path.append(goal)
            print('A* Nodes visited: ', len(visited))
            print('A* Solution nodes expanded: ', len(path))
            return path

        neighbors = list(graph.neighbors(node))

        # Store the order of cities based on the costs
        ordered_neighbors = []
        
        for i, neighbor in enumerate(neighbors):
            if neighbor in visited:
                continue

            # A_Star Queueing strategy
            if i == 0:
                ordered_neighbors.append(neighbor)
            else:

                # Cost to neighbor
                cost = get_distance(node, neighbor)

                # Neighbor's cost to destination
                heuristic_cost = get_manhattan(neighbor, goal)
                neighbor_cost = cost + heuristic_cost

                # Insert the neighbor in the index based on the cost relative to other neighbors
                for j, successor in enumerate(ordered_neighbors):
                    successor_cost = get_distance(node, successor) + get_manhattan(successor, goal)

                    if successor_cost > neighbor_cost:
                        ordered_neighbors.insert(j, neighbor)
                        break
                ordered_neighbors.append(neighbor)

        # Append the ordered neighbors to the queue for node expansion
        for neighbor in ordered_neighbors:
            queue.append(neighbor)
            # Insert parent node for path trace back when goal is found
            parent_dict[neighbor] = node