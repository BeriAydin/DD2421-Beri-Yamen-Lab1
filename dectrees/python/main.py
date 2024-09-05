import monkdata as m
import dtree as d

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
monk_1Sk_5 = d.select(m.monk1, m.attributes[4], 4)   
monk_1Sk_5_1 = d.averageGain(monk_1Sk_5, m.attributes[0])
monk_1Sk_5_2 = d.averageGain(monk_1Sk_5, m.attributes[1])
monk_1Sk_5_3 = d.averageGain(monk_1Sk_5, m.attributes[2])
monk_1Sk_5_4 = d.averageGain(monk_1Sk_5, m.attributes[3])
monk_1Sk_5_6 = d.averageGain(monk_1Sk_5, m.attributes[5])

# sort and print the results
print("MONK-1", sorted([(f"{d.averageGain(monk_1Sk_5, a):.4g}", a) for a in m.attributes], reverse=True))

