"""
Created on May, 2025

@author: Ayoub Wahmane/securityinshadows
"""

import time  # for timing stuff, we'll make it super precise
from collections import deque  # might need this later if we do tree stuff

#  simple tree node class first
class TreeNode:
    def __init__(self, key, freq):
        self.key = key  # the actual value (like 10, 20 etc)
        self.freq = freq  # how often we search for this bad boy
        self.left = None  # left child - starts empty
        self.right = None  # right child - starts empty

# RECURSIVE APPROACH 
# this is the brute force way - checks EVERY possible tree
# warning: gets slow with more than 15 keys
def recursive_obst(keys, freq, i, j):
    """tries all possible roots recursively, not efficient but simple"""  
    # base case 1: if the range is invalid
    if j < i:
        return 0  # cost is zero for empty subtree 
    # base case 2: single node
    if i == j:
        return freq[i]  # just return its own frequency    
    # start with infinity as our initial min cost
    min_cost = float('inf')
    # try every key in current range as root
    for r in range(i, j + 1):
        # cost = left subtree + right subtree + sum of all freqs in this subtree
        # the sum part is because adding a root increases depth of all other nodes
        cost = (recursive_obst(keys, freq, i, r - 1) +
               recursive_obst(keys, freq, r + 1, j) +
               sum(freq[i:j+1]))      
        # keep track of the minimum cost we find
        if cost < min_cost:
            min_cost = cost 
    return min_cost  # return the best cost we found

# DIVIDE AND CONQUER WITH MEMOIZATION / DYNAMIC PROGRAMMING APPROACH
# this is the smart version that stores results to avoid re-calculating
# way faster than recursive for medium-sized inputs
def dp_obst(keys, freq):
    """smarter version using memoization, O(n^3) time"""  
    n = len(keys)
    # dp table where dp[i][j] = min cost for keys i to j
    dp = [[0]*n for _ in range(n)]  
    # sum_cache stores sum(freq[i..j]) so we don't recompute it a million times
    sum_cache = [[0]*n for _ in range(n)]  
    # precompute all the sum ranges - makes things faster
    for i in range(n):
        sum_cache[i][i] = freq[i]  # sum of single element is itself
        for j in range(i+1, n):
            sum_cache[i][j] = sum_cache[i][j-1] + freq[j]  # add next element  
    # fill diagonal - cost for single node is just its frequency
    for i in range(n):
        dp[i][i] = freq[i]  
    # now fill the dp table for all lengths from 2 to n
    for L in range(2, n+1):  # L is length of current segment
        for i in range(n-L+1):  # i is starting index
            j = i + L - 1  # j is ending index
            dp[i][j] = float('inf')  # start with infinity          
            # try all possible roots in current range
            for r in range(i, j+1):
                # cost if we choose r as root
                left = dp[i][r-1] if r > i else 0  # left subtree cost
                right = dp[r+1][j] if r < j else 0  # right subtree cost
                cost = left + right + sum_cache[i][j]  # total cost           
                # keep the minimum cost we find
                if cost < dp[i][j]:
                    dp[i][j] = cost  
    # the answer is in the top-right corner of the table
    return dp[0][n-1]

# GREEDY APPROACH
# this doesn't give optimal solution but runs fast
# sometimes good enough if you need speed over perfection
def greedy_obst(keys, freq):
    """builds a BST by always picking the most frequent key next"""
    
    # first sort keys by frequency (descending)
    nodes = sorted(zip(keys, freq), key=lambda x: -x[1])
    
    # build the BST by inserting nodes in order of frequency
    root = None
    for key, prob in nodes:
        root = insert_greedy(root, key, prob)
    
    # calculate the actual search cost of this tree
    cost = calculate_cost(root)
    return cost, root  # return both cost and tree for inspection

# helper to insert into BST for greedy approach
def insert_greedy(node, key, prob):
    """inserts a node into BST following standard BST rules"""
    if node is None:
        return TreeNode(key, prob)  # found empty spot, make new node
    
    # standard BST insertion - smaller keys go left
    if key < node.key:
        node.left = insert_greedy(node.left, key, prob)
    else:
        node.right = insert_greedy(node.right, key, prob)
    return node

# helper to calculate cost of a built tree
def calculate_cost(root, depth=1):
    """recursively calculates search cost of a given BST"""
    if root is None:
        return 0  # empty tree has zero cost
    
    # current node's contribution + left subtree + right subtree
    return (root.freq * depth +  # current node
            calculate_cost(root.left, depth+1) +  # left kids
            calculate_cost(root.right, depth+1))  # right kids

#TESTING CODE 
def test_obst_algorithms():
    """runs our algorithms on different test cases"""
    
    # some test cases with different patterns
    test_cases = [
    ([10, 20, 30, 40], [0.1, 0.2, 0.4, 0.3], "4 keys: balanced"),
    ([5, 10, 15], [0.2, 0.5, 0.3], "3 keys: middle-heavy"),
    ([1, 2, 3, 4, 5], [0.05, 0.4, 0.08, 0.04, 0.43], "5 keys: skewed"),
    ([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 
     [0.05, 0.1, 0.15, 0.05, 0.2, 0.1, 0.05, 0.15, 0.1, 0.05],
     "10 keys: mixed frequencies"),
    # 250-key test case with Zipf-like distribution (common in real-world data) only prototype (it takes too long to do recursive)
 #   (list(range(1, 251)),
  #   [1.0/(i+10) for i in range(250)],  # Decreasing probabilities
 #    "250 keys: Zipf-like distribution"),
 ]
    
    for keys, freq, description in test_cases:
        print(f"\n{'='*50}")
        print(f"testing: {description}")
        print(f"keys: {keys}")
        print(f"frequencies: {freq} (sum: {sum(freq):.2f})")
        
        # test recursive approach
        start = time.perf_counter()  # high precision timer
        cost_rec = recursive_obst(keys, freq, 0, len(keys)-1)
        rec_time = time.perf_counter() - start
        print(f"\n1. recursive cost: {cost_rec:.4f}")
        print(f"took: {rec_time:.6f} secs")
        
        # test DP approach
        start = time.perf_counter()
        cost_dp = dp_obst(keys, freq)
        dp_time = time.perf_counter() - start
        print(f"\n2. DP cost: {cost_dp:.4f}")
        print(f"took: {dp_time:.6f} secs")
        # using emojis for better visualization
        print(f"check: {'✅' if abs(cost_rec - cost_dp) < 0.001 else '❌'}")
        
        # test greedy approach
        start = time.perf_counter()
        cost_greedy, greedy_tree = greedy_obst(keys, freq)
        greedy_time = time.perf_counter() - start
        print(f"\n3. greedy cost: {cost_greedy:.4f}")
        print(f"took: {greedy_time:.6f} secs")
        print(f"ratio vs optimal: {cost_greedy/cost_dp:.2f}x")

# helper to print tree structure visually
def print_tree(root, level=0, prefix="root: "):
    """prints the BST sideways so we can see the structure"""
    if root is not None:
        print(" " * (level*4) + prefix + str(root.key))
        if root.left is not None or root.right is not None:
            print_tree(root.left, level+1, "L--- ")
            print_tree(root.right, level+1, "R--- ")

# run the tests if this file is executed directly
if __name__ == "__main__":
    print("running optimal BST tests...")
    test_obst_algorithms()