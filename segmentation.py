import string


agents = (0, 0), (0, 1), (2, 2), (4, 3)


class AxisSegmentation:
    def __init__(self, index):
        self.__index = index
        self.__clusters = {}
        
    def append(self, agent):
        a, b = agent
        
        self.__clusters[b] = agent
    
    def sort(self):
        return list(self.__clusters.values())
    
    @property
    def index(self):
        return self.__index
    
    @property
    def clusters(self):
        return self.__clusters
    
    def __iter__(self):
        return iter(self.__clusters)
    
    def __len__(self):
        return len(self.__clusters)


class TreeClassify:
    def __init__(self, size):
        self.agents = []
        self.__segmentation = {}
    
        for c in range(size):
            self.__segmentation[c] = AxisSegmentation(c)
        
    def insert(self, agent):
        self.agents.append(agent)
        
    def sort(self):
        for agent in self.agents:
            a, b = agent
            self.__segmentation[a].append(agent)
        
        
        clusters = []
        group = []
        
        for index, cluster in self.__segmentation.items():
            if len(cluster) <= 0:
                clusters.append(group)
                group = []
            else:
                group += (cluster.sort())
        
        return [*filter(lambda a: len(a) > 0, clusters)]


if __name__ == "__main__":
    tree = TreeClassify(10)

    for agent in agents:
        tree.insert(agent)

    clusters = tree.sort()

    for i, cluster in enumerate(clusters):
        print(f"{i}: {cluster}")
