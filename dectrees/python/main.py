import random
from itertools import count
import monkdata as m
import dtree as d
import drawtree_qt5 as pyqt
import numpy as np
import matplotlib.pyplot as plt



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

monk1tree = d.buildTree(m.monk1, m.attributes)
monk2tree = d.buildTree(m.monk2, m.attributes)
monk3tree = d.buildTree(m.monk3, m.attributes)

err1_train = 1-d.check(monk1tree, m.monk1)
err1_test = 1-d.check(monk1tree, m.monk1test)
err2_train = 1-d.check(monk2tree, m.monk2)
err2_test = 1-d.check(monk2tree, m.monk2test)
err3_train = 1-d.check(monk3tree, m.monk3)
err3_test = 1-d.check(monk3tree, m.monk3test)

print("Error rate for MONK-1 training set:", err1_train)
print("Error rate for MONK-1 test set:", err1_test)
print("Error rate for MONK-2 training set:", err2_train)
print("Error rate for MONK-2 test set:", err2_test)
print("Error rate for MONK-3 training set:", err3_train)
print("Error rate for MONK-3 test set:", err3_test)
print()

#print(monk1tree)
#pyqt.drawTree(monk1tree)

# Assignment 6
# Split the MONK-1 dataset into a training set and a validation set


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruneTree(tree_data, fraction):
    "prunes the tree and returns the pruned tree"
    training_data, validation_data = partition(tree_data, fraction)
    temp_tree = d.buildTree(training_data, m.attributes)
    candidates = d.allPruned(temp_tree)
    
    print("length of candidate list: ",len(candidates))
    # print the error rate for each candidate and the candidate itself using enumerate
    current_max = 0
    for i, candidate in enumerate(candidates):
        temp_rate = d.check(candidate, validation_data)
        #print(f"Correct rate for candidate {i+1}: {temp_rate}")   #log
        if temp_rate >= current_max:
            current_max = temp_rate
            current_max_candidate = candidate
    #print("The pruned tree is: ", current_max_candidate, "with error rate: ", current_max, "and fraction: ", fraction)  #log

    return current_max

pruneTree(m.monk3, 0.6)
print()
#pyqt.drawTree(the_pruned_tree)



# Assignment 7


N = 100
error_arr_monk1 = np.zeros((N, 8))
error_arr_monk3 = np.zeros((N, 8))

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
for fraction in fractions:
    for i in range(N):
        error_arr_monk1[i, fractions.index(fraction)] = pruneTree(m.monk1, fraction)
        error_arr_monk3[i, fractions.index(fraction)] = pruneTree(m.monk3, fraction)

# Calculate the mean and standard deviation of the error rate for each fraction for both datasets
mean_monk1 = np.mean(error_arr_monk1, axis=0)
std_monk1 = np.std(error_arr_monk1, axis=0)
mean_monk3 = np.mean(error_arr_monk3, axis=0)
std_monk3 = np.std(error_arr_monk3, axis=0)

# Print the results in a table for each fraction
for i in range(len(fractions)):
    print(f"Fraction: {fractions[i]} MONK-1 Mean: {mean_monk1[i]} Standard deviation: {std_monk1[i]}")
    print(f"Fraction: {fractions[i]} MONK-3 Mean: {mean_monk3[i]} Standard deviation: {std_monk3[i]}")
    print()

# Plotting
plt.figure(figsize=(10, 8))

# Plotting MONK-1 dataset
plt.errorbar(fractions, mean_monk1, yerr=std_monk1, fmt='o-', color='blue', label='MONK-1')

# Plotting MONK-3 dataset
plt.errorbar(fractions, mean_monk3, yerr=std_monk3, fmt='o-', color='red', label='MONK-3')

# Labeling the axes
plt.xlabel('Fraction')
plt.ylabel('Mean Error')
plt.title('Error vs. Fraction for MONK-1 and MONK-3 Datasets')

# Adding legend to differentiate between different datasets
plt.legend(title="Datasets")

# Show the plot
plt.grid(True)
plt.show()






        
    






