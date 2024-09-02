import monkdata as m
import dtree as d

entrop_monk1 = d.entropy(m.monk1)
entrop_monk2 = d.entropy(m.monk2)
entrop_monk3 = d.entropy(m.monk3)

print(entrop_monk1,entrop_monk2,entrop_monk3)