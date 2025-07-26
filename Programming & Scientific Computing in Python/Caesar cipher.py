# -*- coding: utf-8 -*-




#freqs_en()
#'C:\Python\ch-freq-en.txt'
def freqs_en(addoffre):
    #j=0
    freqs_en=26*[0]
    with open(addoffre,'r') as freq:
            for line in freq.readlines():
                firsttry=line.strip().split('\t')   
                freqs_en[ord(firsttry[0])-ord('A')] = float(firsttry[1])/100.0
                #freqs_en[j]=float(firsttry[1])/100.0
                #j+=1
    freq.close()
    return freqs_en

#'C:\Python\secret_files\secret1.txt'

#getfreq()
def getfreq(text):
    j=0
    i=97
    data=26*[0]
    import re
    text1=re.sub('[^\u0061-\u007a\u0041-\u005a]','',text)
    text1=text1.lower()
    for i in range(97,123):
        j=text1.count(chr(i))
        data[i-97]=j/len(text1)
    return data

#caldiff()
def calcdiff(table1,table2):
    k=0
    diff=26*[0]
    for k in range(0,26):
        diff[k]=abs(table1[k]-table2[k])
    difference=sum(diff)    
    return difference

#cipher()
def cipher(text,shift=0):
    data1=[0]*len(text)
    j=0
    newtext=str()
    for ch in text:
        if 'A' <= ch.upper() <='Z': 
            if 'A' <= ch <='Z': 
                data1[j]=chr((ord(ch)-ord('A')+shift)%26+ord('A'))
            elif 'a'<= ch<='z':
                data1[j]=chr((ord(ch)-ord('a')+shift)%26+ord('a'))
        else:
            data1[j]=ch
        j+=1
    newtext=newtext.join(data1)
    return newtext

#findshift()
def findshift(secret):
    freqs=freqs_en('C:\Python\ch-freq-en.txt')
    optshift=int()
    b=[0]*25
    for k in range(1,26):
        secret1=cipher(secret,k)
        a=getfreq(secret1)
        b[k-1]=calcdiff(a,freqs)
    optshift=b.index(min(b))+1
    return optshift
 

   
address=str(input('Input the address of your encode file:'))
with open(address,'r') as secret0:
    key=findshift(secret0.read())
output=str(input('Input the address of your new decode file:'))
with open(output,'w') as decode:
    with open(address,'r') as secret0:
        for line in secret0:
            secret=line.strip()
            print(cipher(secret,key),file=decode)