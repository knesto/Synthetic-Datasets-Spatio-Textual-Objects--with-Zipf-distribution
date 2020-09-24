#from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import re
from operator import itemgetter 

alpha=1
x = -1 
m_bFirst = True #// Static first time flag
c=0 #Normalization constant

def uniform_dataset(words):
    size = 1000000 # size of records 
    x=np.random.uniform(30.0273437,60.02734375,size) # creation x points in [30.0273437,60.02734375] with uniform distribution
    y=np.random.uniform(35.87531083569679,45.87531083569679,size)
    #visualization(x,y)
    generate_dataset(words,x,y,'uniform_dataset1.txt')
        

def non_uniform_dataset(words):

    clusters = 5 
    size = 200000 # 200000 records per cluster
    
    x_center = [30.02734375,40.02734375,50.02734375,60.02734375,70.02734375] # set x centers

    y_center = [55.87531083569679,45.87531083569679,35.87531083569679,25.87531083569679,15.87531083569679] # set y centers

    scales = [0.25,0.8,0.42,0.11,0.55] # set scale per cluster

    for i in range(clusters): #creation 5 clusters
        if i == 0:
            x = np.random.normal(loc=x_center[i], scale=scales[i], size=size)# creation  the first x point  with normal distribution 
            y = np.random.normal(loc=y_center[i], scale=scales[i], size=size)
            continue
        x = np.hstack([x, np.random.normal(loc=x_center[i], scale=scales[i], size=size)])
        y = np.hstack([y, np.random.normal(loc=y_center[i], scale=scales[i], size=size)])
        
    #visualization(x,y)  
    generate_dataset(words,x,y,'non_uniform_dataset.txt')


def visualization(x,y): # visualization of data on earth map
    
    BBox =(-180, 180,-90,90)
    ruh_m = plt.imread('gen_uni_1.png')

    fig, ax = plt.subplots(figsize = (8,7))
    ax.scatter(x, y, zorder=1, alpha= 0.2, c='b', s=13)
    ax.set_title('Uniform Dataset')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
    plt.show()


def generate_dataset(words,x,y,file): 
    totalkeywordlist=[]
    with open(file, "w") as f:  
      for i in range(len(x)):
        r = int(random.uniform(1, 5))
        keywords=[]
        pin= [0] * r
        nRand= random.random()
        rand_val(nRand)
        for p in range(len(words)-1):
            nZipf = int(nextZipf(r))    
            pin[nZipf]=(pin[nZipf]+1)
        for j in range(len(pin)):
            
            if pin[j] in keywords:
                i=i-1
                break
            keywords.append(words[pin[j]])
            totalkeywordlist.append(words[pin[j]])
                                                     
        finalkeywords = ','.join(keywords)
        
        f.write("2|" + str(x[i]) + "|" + str(y[i]) + "|" + finalkeywords + "\n")
    validation_zipf(totalkeywordlist)


def nextZipf(n):
    global alpha,x,m_bFirst,c
    
    zipf_value = 0
    # Compute normalization constant on first call only
    if (m_bFirst==True):
        i=1
        while i <=n:
            c = c + (1.0 / math.pow(float (i), alpha))
            i=i+1
    c = 1.0 / c;
    m_bFirst = False

    #Pull a uniform random number (0 < z < 1)
    i = 1
    while True:
        z = rand_val(0)
        if((z != 0) or (z !=1)):
            break
        i = i + 1
    
    #Map z to the value
    sum_prob = 0
    i=1
    while i <=n:
        sum_prob = sum_prob+c / math.pow(float (i), alpha)
        if sum_prob>= z:
            zipf_value = i
            break
        i=i+1
    return (int (zipf_value)-1)

"""//=========================================================================
//	= Multiplicative LCG for generating uniform(0.0, 1.0) random numbers    =
//	=   - x_n = 7^5*x_(n-1)mod(2^31 - 1)                                    =
//	=   - With x seeded to 1 the 10000th x value should be 1043618065       =
//	=   - From R. Jain, "The Art of Computer Systems Performance Analysis," =
//	=     John Wiley & Sons, 1991. (Page 443, Figure 26.2)                  =
//========================================================================="""

def rand_val(seed):
    global x
    a = 16807 #Multiplier
    m = 2147483647 #Modulus
    q = 127773 #m div a
    r = 2836 #m mod a
    
    #Set the seed if argument is non-zero and then return zero
    if (seed > 0):
        x = seed
        return (0.0)
    
    #RNG using integer arithmetic
    x_div_q = x / q #x divided by q
    x_mod_q = x % q #x modulo q
    x_new = (a * x_mod_q) - (r * x_div_q) #New x value

    if (x_new > 0):
        x = x_new
    else:
      x = x_new + m
    # Return a random value between 0.0 and 1.0
    return (float (x) / m)


def validation_zipf(keywordlist):
    wordfreq = []
    for w in keywordlist:
        wordfreq.append(keywordlist.count(w))
    #print("Pairs\n" + str(list(zip(keywordlist, wordfreq))))
    with open("zipf_validation.txt", "w") as f:
        f.write(str(list(zip(keywordlist, wordfreq))))
def main():
    words=[]
    file = open("keywords.txt","r",encoding='utf-8')
    for line in file:
        fields = line.split("\n")
        words.append(fields[0])
        
    uniform_dataset(words)
    non_uniform_dataset(words)
    

if __name__=="__main__":
    main()

