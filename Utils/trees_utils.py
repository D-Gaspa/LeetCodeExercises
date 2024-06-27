import tempfile

from graphviz import Digraph


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeVisualizer:
    @staticmethod
    def visualize(root: TreeNode) -> str:
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

        # Generate a unique file name and save the file in the current directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='', dir='.') as tmp:
            image_path = tmp.name
            tmp.close()  # Close the file before rendering
            dot.render(image_path, format='png', cleanup=True)

        return image_path
