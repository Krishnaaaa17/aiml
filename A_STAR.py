def astar(start, goal, graph):
    open_set = {start}
    closed_set = set()
    g = {start: 0}
    parents = {start: start}
    
    def heuristic(node):
        H_dist = {
            'A': 10, 'B': 8, 'C': 5, 'D': 7, 'E': 3,
            'F': 6, 'G': 5, 'H': 4, 'I': 1, 'J': 0
        }
        return H_dist.get(node, float('inf'))
    
    def get_neighbors(node):
        return graph.get(node, [])
    
    while open_set:
        current = min(open_set, key=lambda x: g[x] + heuristic(x))
        
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = parents[current]
            path.append(start)
            path.reverse()
            return path
        
        open_set.remove(current)
        closed_set.add(current)
        
        for neighbor, weight in get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            tentative_g = g[current] + weight
            
            if neighbor not in open_set or tentative_g < g.get(neighbor, float('inf')):
                parents[neighbor] = current
                g[neighbor] = tentative_g
                open_set.add(neighbor)
    
    return None

# Example usage:
graph = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)]
}

path = astar('A', 'J', graph)
if path:
    print("Path Found:", path)
else:
    print("Path Doesn't Exist!")
    
    