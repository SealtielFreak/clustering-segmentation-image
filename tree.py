class NodeContainer:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class TreeSegmentation:
    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = NodeContainer(value)
        
        if self.root is None:
            self.root = new_node
        else:
            current = self.root
            
            while True:
                if value <= current.value:
                    if current.left is None:
                        current.left = new_node
                        break
                    else:
                        current = current.left
                else:
                    if current.right is None:
                        current.right = new_node
                        break
                    else:
                        current = current.right

    def in_order_traversal(self, node, result):
        if node:
            self.in_order_traversal(node.left, result)
            result.append(node.value)
            
            self.in_order_traversal(node.right, result)

    def sort(self, arr):
        for value in arr:
            self.insert(value)
            
        sorted_arr = []
        self.in_order_traversal(self.root, sorted_arr)
        
        return sorted_arr


if __name__ == "__main__":        
    tree = TreeSegmentation()
    arr = [5, 2, 8, 1, 9, 4]
    sorted_arr = tree.sort(arr)
    
    print(sorted_arr)
    