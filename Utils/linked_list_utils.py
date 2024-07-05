from graphviz import Digraph


class ListNode:
    def __init__(self, val=0, next_node=None):
        self.val = val
        self.next_node = next_node


class LinkedListVisualizer:
    @staticmethod
    def visualize(head: ListNode, file_name: str) -> str:
        dot = Digraph(comment='Linked List')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='record', style='filled', color='lightblue', fontcolor='black')
        dot.attr('edge', color='black')

        current = head
        while current:
            node_id = str(id(current))
            dot.node(node_id, str(current.val))
            if current.next_node:
                dot.edge(node_id, str(id(current.next_node)))
            current = current.next_node

        # Use the provided file name directly
        dot.render(file_name, format='png', cleanup=True)

        return file_name + '.png'
