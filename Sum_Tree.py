import numpy as np

class Sum_Tree:
    def __init__(self,l):
        self.leafs = l # number of leaf nodes
        self.tree_nodes = np.zeros(2*self.leafs - 1) # total number of nodes in tree
        self.priorities = np.zeros(l) # priority data

    # updates value of a single leaf and calls for tree update
    def update_leaf(self,leaf_no,leaf_value):
        self.priorities[leaf_no] = leaf_value

        tree_index = self.leafs-1+leaf_no
        leaf_change = leaf_value - self.tree_nodes[tree_index]
        self.tree_nodes[tree_index] = leaf_value

        self.update_tree(tree_index,leaf_change)

    # propagates the change in leaf value up the tree
    def update_tree(self,tree_index,delta):
        parent_node = (tree_index-1)//2

        self.tree_nodes[parent_node]+=delta

        if parent_node!=0:
            self.update_tree(parent_node,delta)

    # gets priority for the given sum value
    def get_priority(self,sum,index):

        left_node = 2*index + 1
        right_node = left_node + 1

        if left_node>=len(self.tree_nodes):
            return index-self.leafs+1,self.tree_nodes[index]

        if sum<self.tree_nodes[left_node]:
            return self.get_priority(sum,left_node)
        else:
            return self.get_priority(sum-self.tree_nodes[left_node],right_node)

    # updates the tree based on current priority data
    def update_allLeaves(self):
        for leaf_no,leaf_value in enumerate(self.priorities):
            self.update_leaf(leaf_no,leaf_value)




