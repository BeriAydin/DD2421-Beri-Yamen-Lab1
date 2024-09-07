from itertools import count
import monkdata as m
import dtree as d
import drawtree_qt4 as qt4

# Assignment 1
# Calculate the entropy of the training datasets
print("Entropy of MONK-1:", d.entropy(m.monk1))
print("Entropy of MONK-2:", d.entropy(m.monk2))
print("Entropy of MONK-3:", d.entropy(m.monk3))
print()

# Assignment 2 use averageGain function
# MONK-1
avarage_gain1_1 = d.averageGain(m.monk1, m.attributes[0])
avarage_gain1_2 = d.averageGain(m.monk1, m.attributes[1])
avarage_gain1_3 = d.averageGain(m.monk1, m.attributes[2])
avarage_gain1_4 = d.averageGain(m.monk1, m.attributes[3])
avarage_gain1_5 = d.averageGain(m.monk1, m.attributes[4])
avarage_gain1_6 = d.averageGain(m.monk1, m.attributes[5])
# MONK-2 
avarage_gain2_1 = d.averageGain(m.monk2, m.attributes[0])   
avarage_gain2_2 = d.averageGain(m.monk2, m.attributes[1])
avarage_gain2_3 = d.averageGain(m.monk2, m.attributes[2])
avarage_gain2_4 = d.averageGain(m.monk2, m.attributes[3])
avarage_gain2_5 = d.averageGain(m.monk2, m.attributes[4])
avarage_gain2_6 = d.averageGain(m.monk2, m.attributes[5])
# MONK-3
avarage_gain3_1 = d.averageGain(m.monk3, m.attributes[0])
avarage_gain3_2 = d.averageGain(m.monk3, m.attributes[1])
avarage_gain3_3 = d.averageGain(m.monk3, m.attributes[2])
avarage_gain3_4 = d.averageGain(m.monk3, m.attributes[3])
avarage_gain3_5 = d.averageGain(m.monk3, m.attributes[4])
avarage_gain3_6 = d.averageGain(m.monk3, m.attributes[5])
# present the results in a table with 5 significant digits and sorted by the information gain
print("MONK-1", sorted([(f"{d.averageGain(m.monk1, a):.4g}", a) for a in m.attributes], reverse=True))
print("MONK-2", sorted([(f"{d.averageGain(m.monk2, a):.4g}", a) for a in m.attributes], reverse=True))
print("MONK-3", sorted([(f"{d.averageGain(m.monk3, a):.4g}", a) for a in m.attributes], reverse=True))
print()

# Assignment 5

# level 1: strat splittingt the MONK-1 dataset with A5
monk1_5_1 = d.select(m.monk1, m.attributes[4], 1)
monk1_5_2 = d.select(m.monk1, m.attributes[4], 2)
monk1_5_3 = d.select(m.monk1, m.attributes[4], 3)
monk1_5_4 = d.select(m.monk1, m.attributes[4], 4)

# calculate the entropy of the subsets of level 1
print("Entropy of MONK-1 A5=1:", d.entropy(monk1_5_1))
print("Entropy of MONK-1 A5=2:", d.entropy(monk1_5_2))
print("Entropy of MONK-1 A5=3:", d.entropy(monk1_5_3))
print("Entropy of MONK-1 A5=4:", d.entropy(monk1_5_4))
print()

# level 2: strat splitting the MONK-1>>a5 nodes. The first node (a5=1) is now a leaf node and we will only split the other nodes
# choose the attribute with the highest information gain for the node where a5=2
print("MONK-1 A5=2", sorted([(f"{d.averageGain(monk1_5_2, a):.4g}", a) for a in m.attributes], reverse=True))
# choose the attribute with the highest information gain for the node where a5=3
print("MONK-1 A5=3", sorted([(f"{d.averageGain(monk1_5_3, a):.4g}", a) for a in m.attributes], reverse=True))
# choose the attribute with the highest information gain for the node where a5=4
print("MONK-1 A5=4", sorted([(f"{d.averageGain(monk1_5_4, a):.4g}", a) for a in m.attributes], reverse=True))
print()

tree = d.buildTree(m.monk1, m.attributes)
print(tree)
qt4.drawTree(tree)


