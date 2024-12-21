#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import numpy
import unicodecsv
import random
import operator
import math
import string
import re
from collections import OrderedDict

#Lets read the files first
with open('data/train.dat') as train:
    train_lines = train.readlines()

with open('data/test.dat') as test:
    test_lines = test.readlines()   

#Split the files
docs_train = [l.split() for l in train_lines] 
docs_test = [l.split() for l in test_lines]



for i in range(25000):
    docs_train[i] = set(docs_train[i])
    docs_test[i] = set(docs_test[i])
    
#COnversion into lower case
s=1
while s < 25000:
    docs_train[s] = [w.lower() for w in docs_train[s]]
    docs_test[s] = [w.lower() for w in docs_test[s]]
    s = s+1


#Loop for removing commas and other things from test data
for count in range(0, 24999):   
    docs_test[count] = [s.replace(',' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace(',' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('(' , ' ') for s in docs_test[count]]
    docs_test[count] = [s.replace(')' , ' ') for s in docs_test[count]]
    docs_test[count] = [s.replace("'" , ' ') for s in docs_test[count]]
    docs_test[count] = [s.replace('"' , ' ') for s in docs_test[count]]
    docs_test[count] = [s.replace('-' , ' ') for s in docs_test[count]]
    docs_test[count] = [s.replace('.' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('br' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('!' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('<' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('>' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace(' ' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('[' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace(']' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('{' , '') for s in docs_test[count]]
    docs_test[count] = [s.replace('}' , '') for s in docs_test[count]]
    count = count + 1
    

#Loop for removing commas and other things from train data
for count in range(0, 24999):   
    docs_train[count] = [s.replace(',' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace(',' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('(' , ' ') for s in docs_train[count]]
    docs_train[count] = [s.replace(')' , ' ') for s in docs_train[count]]
    docs_train[count] = [s.replace("'" , ' ') for s in docs_train[count]]
    docs_train[count] = [s.replace('"' , ' ') for s in docs_train[count]]
    docs_train[count] = [s.replace('-' , ' ') for s in docs_train[count]]
    docs_train[count] = [s.replace('.' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('br' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('!' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('<' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('>' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace(' ' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('/' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace("{" , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('1' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('-1' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('}' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace(']' , '') for s in docs_train[count]]
    docs_train[count] = [s.replace('{' , '') for s in docs_train[count]]
    count = count + 1

    
dict1 = {}
#lol = [] 
lin = []
r_final = []
with open('writing.dat', 'w') as writing:
    for count in range(0, 25000):
        
        for count1 in range(0,25000):
            #This is the code for intersection
           
            intersection = len(list(set(docs_test[count]) & set(docs_train[count1])))
            #print intersection
            #This is the code for union
            union = len(list(set(docs_test[count]) | set(docs_test[count1])))

            #conversion of int to float
            intersection = int(intersection)
            a = float(intersection)
            o = float(union)
            #print intersection
            #print union
            #Calculation of Jaccard similarity
            m = a / o

            #Conversion into float again
            float(m)

            #saving jacard similiarity(m) into list
            #lol.append({count1 : m})
            dict1[count1] = m

            #When we need to exit the inner loop, lets calculate the rating
            if count1 == 24999:
               #Making the ordered dictionary
                d_s = OrderedDict(sorted(dict1.items(), key=lambda x: x[1]))
                #print d_s 
                #Taking the last three entries and saving into j1, j2, j3
                j1 = d_s.keys()[-1]
                j2 = d_s.keys()[-2]
                j3 = d_s.keys()[-3]
                #print j3
                j4 = d_s.keys()[-4]
                #print j4
                j5 = d_s.keys()[-5]
                #print j5

                #Getting the rating from the j1, j2, j3 and saving it into line, line1 , line2
                line = train_lines[j1].split()
                if line[0] == '+1':
                    line = 1
                else:
                    line = 0

                line1 = train_lines[j2].split()

                if line1[0] == '+1':
                    line1 = 1
                else:
                    line1 = 0                

                line2 = train_lines[j3].split()
                
                if line2[0] == '+1':
                    line2 = 1
                else:
                    line2 = 0
                    
                line3 = train_lines[j4].split()
                if line3[0] == '+1':
                    line3 = 1
                else:
                    line3 = 0
                
                line4 = train_lines[j5].split()
                if line4[0] == '+1':
                    line4 = 1
                else:
                    line4 = 0

                #Calculate the rating from the line, line1, line2
                r_total = line + line1 + line2 + line3 + line4
    
                if r_total <= 2:
                    #print ("So the rating for %d is NEGATIVE" % count)
                    rating_final = str("-1")   
                    writing.write('-1\n')
                else:    
                    #print ("So the rating for %d is POSITIVE" % count)
                    rating_final = str("+1")
                    writing.write('+1\n')
                d_s.clear()
                #OrderedDict.clear()
                dict1.clear()


# In[ ]:




