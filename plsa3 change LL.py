import random
import os
import sys
import math



inpath = "D://python2.7.6//MachineLearning//plsa//allfiles-topic model-1"
outfile1 = "D://python2.7.6//MachineLearning//plsa//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//plsa//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//plsa//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//plsa//4.txt"
     
numZ=5
iterNum=200
classList=["business","auto","sport","it","yule"]
eps=0.00001
######################

def loadData():
    global docDic;global wordDic;
    

    ##############word dictionary for each doc
    wordDic={}
    docDic={}
    i=0
    for filename in os.listdir(inpath):
        docDic[i]=[{},'label']
        #i+=1
        #######label
        for c in classList:
            if filename.find(c)!=-1:
                docDic[i][1]=c
        #######wordDic         
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for wid in words:
            if len(wid.strip())<=2:continue

            if wid not in wordDic:
                wordDic[wid]={}
                docDic[i][0][wid]=1
            if wid in wordDic:
                if wid not in docDic[i][0]:
                    docDic[i][0][wid]=1
                    
                if wid in docDic[i][0]:
                    docDic[i][0][wid]+=1
        #######
        i+=1
    print '%d doc, %d wid loaded'%(i,len(wordDic))
    ######################loaded

    outPutfile=open(outfile1,'w')
    for did,wl in docDic.items():
        outPutfile.write(str(did));
        outPutfile.write(':')
        outPutfile.write(str(wl))
        outPutfile.write('\n')
    outPutfile.close()



def entropy():
    global docDic;global wordDic;global featList ;featList=[]
    for feat in wordDic.keys():
        #######count and fill {feat:{it:0,busi:0...}
        for c in classList:
            wordDic[feat][c]=0.0
            ####to count each feat each c:start go through doc
            for did,wl in docDic.items():
                if wl[1]==c and feat in wl[0].keys():
                    wordDic[feat][c]+=docDic[did][0][feat]
        #print wordDic[feat]#####all doc through
        wordDic[feat]['entropy']=0.0
        ####### count sum
        for c in classList:
            wordDic[feat]['entropy']+= wordDic[feat][c]
        #print wordDic[feat]
        ######## in one kind doc how many docs include feat/in all doc how many doc include feat
        for c in classList:
            wordDic[feat][c]/=wordDic[feat]['entropy']
        #print wordDic[feat]
        ########-sum[p*log(p)]
        wordDic[feat]['entropy']=0.0
        for c in classList:
            #wordDic[feat]['entropy']-=wordDic[feat][c]*math.log(eps+wordDic[feat][c])   wrong
            #cannot add eps in log: (prob+eps) may>1,cannot be probability that<=1,log(p+eps may>1)=+,entropy=-
            if wordDic[feat][c]>0.0:
                wordDic[feat]['entropy']-=wordDic[feat][c]*math.log(wordDic[feat][c])
    #################output
    outPutfile=open(outfile2,'w')
    for feat,dic in wordDic.items():
        outPutfile.write(str(feat));
        outPutfile.write(':')
        outPutfile.write(str(dic))
        outPutfile.write('\n')
    outPutfile.close()      
    ##########compare to get (min,max) entropy interval
    minEn=None;maxEn=None
    for feat in wordDic.keys():
        entropy=wordDic[feat]['entropy']
        if minEn==None or minEn>entropy:
            minEn=entropy
        if maxEn==None or maxEn<entropy:
            maxEn=entropy
    ###########boundary max entropy,copy feat into featlist
    boundary=minEn+(maxEn-minEn)/10000.0
    for feat in wordDic:
        if wordDic[feat]['entropy']==minEn:  ###<boundary???
            featList.append(feat);
    
    print len(wordDic),len(featList),'feat'
            
            
def openSpace():
    global docDic;global wordDic;global featList ;
    global pzfenmu;global pzdfenzi;global pzdfenmu;global pwzfenzi;global pwzfenmu#fenzi fenmu
    pzfenmu={}
    for did in docDic:
        pzfenmu[did]={}
        for wid in featList:
            pzfenmu[did][wid]=0.0
    pzdfenzi={}
    for did in docDic:
        pzdfenzi[did]={}
        for z in range(numZ):
            pzdfenzi[did][z]=0.0
    pzdfenmu={}
    for did in docDic:
        pzdfenmu[did]=0.0
    pwzfenzi={}
    for wid in featList:
        pwzfenzi[wid]={}
        for z in range(numZ):
            pwzfenzi[wid][z]=0.0
    pwzfenmu={}
    for z in range(numZ):
        pwzfenmu[z]=0.0

def initialPara():
    global docDic;global wordDic;global featList ;
    global pzd;global pwz
    pzd={}
    for did in docDic:
        pzd[did]={}
        for z in range(numZ):
            pzd[did][z]=random.uniform(0,1)
    ########use prior knowledge that d1=it,z1=it
    pzd[0][4]=1
    ########count fenmu(sum z) and divide fenmu
    for did in docDic:
        fenmu=0.0
        for z in range(numZ):
            fenmu+=pzd[did][z]
        for z in range(numZ):
            pzd[did][z]/=fenmu
    #print 'pzd',pzd################################
    pwz={}
    for wid in featList:
        pwz[wid]={}
        for z in range(numZ):
            pwz[wid][z]=random.uniform(0,1)
    #######count fenmu(sum w) and divide fenmu
    for z in range(numZ):
        fenmu=0.0
        for wid in featList:
            fenmu+=pwz[wid][z]
        for wid in featList:
            pwz[wid][z]/=(fenmu)
    #print 'pwz',pwz

def EM():
    global docDic;global wordDic;global featList ; #initial once
    global pzd;global pwz
    global pzfenmu;global pzdfenzi;global pzdfenmu;global pwzfenzi;global pwzfenmu #back to zero each time before calc p(z|dw)
    LL=0.0
    ########E step : calc p(z|dw)fenmu
    for d in docDic:
        for w in featList:
            for z in range(numZ):
                pzfenmu[d][w]+=pzd[d][z]*pwz[w][z]/float(len(docDic))
    ########calc p(z|dw)
    for d in docDic:
        for w in featList:
            if w in docDic[d][0]:
                LL+=math.log(pzfenmu[d][w])*docDic[d][0][w]######+eps
                for z in range(numZ):
                    pzdw=pzd[d][z]*pwz[w][z]/(pzfenmu[d][w])######+eps?
                    ########M step : already get posterior prob p(z|dw), now update Loglikely and parameter,all fenzi fenmu accumlated from ZERO 
                    pzdfenzi[d][z]+=pzdw*float(docDic[d][0][w])
                    pzdfenmu[d]+=pzdfenzi[d][z]
                    pwzfenzi[w][z]+=pzdw*float(docDic[d][0][w])
                    pwzfenmu[z]+=pwzfenzi[w][z]
    ###########finish update parameter
    for d in docDic:
        for z in range(numZ):
            pzd[d][z]=pzdfenzi[d][z]/(pzdfenmu[d])#####+eps if add eps, LL not converge in one direction, will fluctuate
    for w in featList:
        for z in range(numZ):
            pwz[w][z]=pwzfenzi[w][z]/(pwzfenmu[z])#######+eps
    ##########
    
    return LL
    

def showBehindTopic():
    global docDic;global wordDic;global featList ; #initial once
    global pzd;global pwz
    topMatric={}
    for z in range(numZ):
        topicWid={};topList=[]#print 'topic %d'%z
        for w in featList:
            topicWid[w]=pwz[w][z]
        topicW=sorted(topicWid.iteritems(),key=lambda a:a[1],reverse=True)  #return [(wid,prob),()...]
        for i in range(10):
            topList.append(topicW[i][0])#print topicW[i][0]
        topMatric[z]=topList
    ######################################
    outPutfile=open(outfile3,'w')
    for z,words in topMatric.items():
        outPutfile.write(str(z));
        outPutfile.write(':')
        for w in words:
            outPutfile.write(str(w))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
            
        
    
 
        
#########################main
loadData()  #go through doc and into file txt
entropy()
openSpace()    
initialPara()

LL1=EM()
LL0=0.0
i=0
while abs(LL1-LL0)>1 and i<iterNum:
    i+=1
    LL0=LL1
    openSpace()
    LL1=EM()
    print 'loglikely',LL1,'difference',LL1-LL0
    


showBehindTopic()

 

        
 
                
        
        
    
    
    







    
    
