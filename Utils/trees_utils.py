from graphviz import Digraph


class BinaryTreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinaryTreeVisualizer:
    @staticmethod
    def visualize(root: BinaryTreeNode, file_name: str) -> str:
        dot = Digraph(comment='Binary Tree')
        dot.attr('node', shape='circle', style='filled', color='lightblue', fontcolor='black')
        dot.attr('edge', color='black')

        def add_nodes_edges(node, parent_id=None):
            if node:
                node_id = str(id(node))
                dot.node(node_id, str(node.val))
                if parent_id:
                    dot.edge(parent_id, node_id)
                add_nodes_edges(node.left, node_id)
                add_nodes_edges(node.right, node_id)

        add_nodes_edges(root)

        # Use the provided file name directly
        dot.render(file_name, format='png', cleanup=True)

        return file_name + '.png'
