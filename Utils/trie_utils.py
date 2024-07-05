from graphviz import Digraph


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False


class TrieVisualizer:
    @staticmethod
    def visualize(root: TrieNode, file_name: str) -> str:
        dot = Digraph(comment='Trie')
        dot.attr(rankdir='LR')  # Makes the graph layout left-to-right
        dot.attr('node', shape='circle', style='filled', color='lightblue', fontcolor='black')
        dot.attr('edge', color='black')

        def add_nodes_edges(node, prefix=''):
            node_id = str(id(node))
            label = 'root' if prefix == '' else prefix[-1]
            if node.is_word_end:
                dot.node(node_id, label, shape='doublecircle')
            else:
                dot.node(node_id, label)

            for char, child in node.children.items():
                child_id = str(id(child))
                dot.edge(node_id, child_id)
                add_nodes_edges(child, prefix + char)

        add_nodes_edges(root)

        # Use the provided file name directly
        dot.render(file_name, format='png', cleanup=True)

        return file_name + '.png'
