import random
import os
import sys
import math



inpath = "D://python2.7.6//MachineLearning//plsa//allfiles-topic model-0"
outfile1 = "D://python2.7.6//MachineLearning//plsa//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//plsa//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//plsa//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//plsa//4.txt"
outfile5 = "D://python2.7.6//MachineLearning//plsa//5.txt"
     
numZ=5
iterNum=200
classList=["business","auto","sport","it","yule"]
alpha=0.1
beta=0.1
######################

def loadData():
    global docList;global wordList;global W;global WW;global D
    

    ##############  for each doc
    docList=[];wordList=[]
    i=0
    for filename in os.listdir(inpath):
        #######label
        for c in classList:
            if filename.find(c)!=-1:
                label=c
        #######wList,zList
        wList=[];zList=[]
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for wid in words:
            if len(wid.strip())<2:continue
            #elif len(wid.strip())>=2:

            if wid not in wordList:
                wordList.append(wid)

            wList.append(wid)
            zList.append(-1)
            i+=1
        #######
        docList.append((wList,zList,label))
    WW=i
    W=len(wordList)
    D=len(docList)
    ##########################################
   
    print '%d doc, %d wid loaded'%(len(docList),len(wordList))
    ######################loaded

    outPutfile=open(outfile1,'w')
    for doc in docList:
        for i in range(len(doc[0])):
            outPutfile.write(str(doc[0][i]))
            outPutfile.write(str(doc[1][i]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    
    outPutfile=open(outfile2,'w')
    for word in wordList:
        outPutfile.write(str(word));
        outPutfile.write('\n')
    outPutfile.close()


def initial():
    global docList;global wordList;global W;global WW;global D
    global nwz;global ndz;global nz;global nd
    nz={};nwz={};nd={};ndz={}
    for k in range(numZ):
        nz[k]=0
        nwz[k]={}
        ndz[k]={}
        for wid in wordList:#'sfds' 'sffe'
            nwz[k][wid]=0
        for d in range(D):
            ndz[k][d]=0
    for d in range(D):
        nd[d]=0
    #################random initial topic for each w each doc
    for doc in docList:
        did=docList.index(doc)#0,1,2,3...
        for i in range(len(doc[0])): #0,1,2,,,
            tempz=random.randint(0,numZ-1)#k=3:0 1 2
            #print word,tempz;
            #wid=doc[0].index(word)#0,1,2,... wrong :the second , will not be given tempz
            
            doc[1][i]=tempz
            ###########count
            nwz[tempz][doc[0][i]]+=1
            ndz[tempz][did]+=1
            nz[tempz]+=1
            nd[did]+=1
    ##############
    outPutfile=open(outfile3,'w')
    for k,v in nwz.items():
        outPutfile.write(str(k));
        outPutfile.write('\n')
        for w,z in v.items():
            outPutfile.write(str(w));
            outPutfile.write(str(z));
            outPutfile.write(' ');
        outPutfile.write('\n')
    outPutfile.close()
    ####
    outPutfile=open(outfile4,'w')
    for k,v in ndz.items():
        outPutfile.write(str(k));
        outPutfile.write('\n')
        for d,z in v.items():
            outPutfile.write(str(d));
            outPutfile.write(':');
            outPutfile.write(str(z));
            outPutfile.write(' ');
        outPutfile.write('\n')
    outPutfile.close()
    ####
    outPutfile=open(outfile5,'w')
    for doc in docList:
        for i in range(len(doc[0])):
            outPutfile.write(str(doc[0][i]))
            outPutfile.write(str(doc[1][i]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    
def onceAllDoc():
    #for each doc each w: nwz ndz nz minus 1, calc p(zi|z-i,w)

    #for each doc each w:sample a topic and pass new topic to each word in each doc

    #for each doc each w: nwz ndz nz(corresponded topic) plus 1,count new :nwz ndz nz nd,word by word

    #after whole doc and word go through once,loglikely

    #after 1000times through all doc ,nwz ndz means something statistically and calc parameter p(w|z) p(d|z)
    global docList;global wordList;global W;global WW;global D
    global nwz;global ndz;global nz;global nd

    wbeta=beta*W
    talpha=alpha*numZ

    for doc in docList:
        did=docList.index(doc)
        for i in range(len(doc[0])): #word index
            #######
            pz=[0.0]*numZ
            wid=doc[0][i]#string
            preTopic=doc[1][i]
            nwz[preTopic][wid]-=1
            ndz[preTopic][did]-=1
            nz[preTopic]-=1
            ########calc p(zi|z-i,w)
            for k in range(numZ):##0,1,2 if numZ=3
                pz[k]=(nwz[k][wid]+beta)*(ndz[k][did]+alpha)/(nz[k]+wbeta)/(nd[did]+talpha)
                if k!=0:
                    pz[k]=pz[k-1]+pz[k]
            #####normalize
            #print '1' ,pz # sum not ==1
            fenmu=pz[numZ-1]
            for k in range(numZ):
                pz[k]/=fenmu
            #print '2', pz
            ######toss and newZ to doclist,nwz ndz nz,loglikely
            toss=random.uniform(0,1)
            for k in range(numZ):
                if toss<=pz[k]:
                    newZ=k;break
            doc[1][i]=newZ
            nwz[newZ][wid]+=1
            ndz[newZ][did]+=1
            nz[newZ]+=1
    ###################log likely
    ll=0.0
    for doc in docList:
        did=docList.index(doc)#0,1,2,3...
        for i in range(len(doc[0])):
            wid=doc[0][i] #string
            pxi=0.0
            for k in range(numZ):
                pxi+=(nwz[k][wid]+beta)*(ndz[k][did]+alpha)/(nz[k]+wbeta)/(nd[did]+talpha)
            ll+=math.log(pxi)
    print 'll',ll
                
            
            
    

#########################main
loadData()
initial()
for i in range(10):
    onceAllDoc()
 

        
 
                
        
        
    
    
    







    
    
