import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def assign_number(strlist: list):
    hashmap = {}
    reverse = {}
    value = 0
    for item in strlist:
        #print(item)
        if item not in hashmap and item != '':
            value = value + 1
            hashmap[item] = value
            reverse[value] = item
    return hashmap,reverse
