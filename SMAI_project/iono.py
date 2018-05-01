
#Imports

from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import csv
import sklearn.preprocessing as pre
import matplotlib.pyplot as mp

#Global variables:
gamma = 10**(-5) # the value of Gamma should be between 0 and 10^(-10)
lam = 10**(2)
mu = 0.1

#----------------------------------------------------------------------
def identity(n):
	m=[[0.0 for x in range(n)] for y in range(n)]
	for i in range(0,n):
		m[i][i] = 1.0
	return m

#----------------------------------------------------------------------


def ones(n,m):
	m=[[1.0 for x in range(m)] for y in range(n)]
	return m


#----------------------------------------------------------------------


def readCSV(data):
	dataList=[]
	classList=[]
	with open(data, "rb") as f_obj:
		reader = csv.reader(f_obj)
		for row in reader:
			if '?' in row:
				continue
			classList.append(row[-1])
			row = row[:-1]
			row = map(float,row)	
			dataList.append(row)
		
	c = len(np.unique(classList))
	return dataList,classList,c



#----------------------------------------------------------------------


def bclsALM(Data_ori,y0,mu):
	epocs = 120
	n = len(Data_ori)
	d = len(Data_ori[0])
	onesArray = np.array(ones(d,d))

	for i in range(0,d):
		for j in range(0,d):
			onesArray[i][j] = (float(onesArray[i][j]))/d
	H = np.array(identity(d)) - np.array(onesArray)
	H = np.array(H)
	tempDataOrig = np.dot(Data_ori,H)
	tempDataOrig = np.array(tempDataOrig)


	c     = len(y0[0])    # 2nd dimension of y0 => no of clusters
	lamda = np.zeros((d,c))
	rho   = 1.005

	idMatrixOfSizeN = identity(n)
	idMatrixOfSizeN = np.array(idMatrixOfSizeN)
	idMatrixOfSizeNU = idMatrixOfSizeN
	for i in range(0,n):
		for j in range(0,n):
			idMatrixOfSizeN[i][j] = (float(idMatrixOfSizeN[i][j]))*gamma

	p = np.linalg.inv(np.add(np.dot(tempDataOrig,tempDataOrig.T),idMatrixOfSizeN))

	objective = []

	for i in range(epocs):
		# Solve W and b
		w = np.dot(np.dot(p,tempDataOrig),y0)
		b = np.mean(y0,axis=0)
		b = np.array(b)[np.newaxis]
		E = np.subtract(np.add(np.dot(tempDataOrig.T,w),np.dot(np.array(ones(d,1)),b)),y0)
	# 	# Solve Z
	# 	# Z = (-2*lam*ones(n)+(mu+2*n*lam)*eye(n))/(mu^2+2*n*lam*mu)*(mu*Y+Lambda); % new solution - O(n^2)
		onesLam = np.array(ones(d,d))
		for out1 in range(0,d):
			for out2 in range(0,d):
				onesLam[out1][out2] = (float(onesLam[out1][out2]))*lam*(2)
		
		identityLam = np.array(identity(d))

		for out1 in range(0,d):
			for out2 in range(0,d):
				identityLam[out1][out2] = (float(identityLam[out1][out2]))*(mu)

		muY0 = []
		for out1 in range(0,d):
			localList = []
			for out2 in range(0,len(y0[0])):
				localList.append(mu*y0[out1][out2])
			muY0.append(localList)

		muY0 = np.array(muY0)
		muY0 = np.add(muY0,lamda)

		a = np.add(onesLam,identityLam)
		a = np.array(a)

		Z = np.linalg.solve(a,muY0)

	# 	# Solve Y
		muZ = []
		for out1 in range(0,len(Z)):
			localList = []
			for out2 in range(0,len(Z[0])):
				localList.append(mu*Z[out1][out2])
			muZ.append(localList)
		muZ = np.array(muZ)

		muZLambda = np.subtract(muZ,lamda)

		twoX = []
		for out1 in range(0,len(tempDataOrig)):
			localList = []
			for out2 in range(0,len(tempDataOrig[0])):
				localList.append(2*tempDataOrig[out1][out2])
			twoX.append(localList)
		

		twoX = np.array(twoX)

		onesB = np.dot(np.array(ones(d,1)),b)
		onesB = np.array(onesB)

		for out1 in range(0,len(onesB)):
			for out2 in range(0,len(onesB[0])):
				onesB[out1][out2] *= 2

		# print np.add((np.dot(twoX.T,w)),onesB)
		tempV = np.add(np.add((np.dot(twoX.T,w)),onesB),muZLambda)
		tempV = np.array(tempV)

		V = []
		for out1 in range(0,len(tempV)):
			localList = []
			for out2 in range(0,len(tempV[0])):
				localList.append((1/(2+mu))*tempV[out1][out2])
			V.append(localList)
		V = np.array(V)

		ind = np.argmax(V, axis=1)  # find index of max value row-wise

		y0 = np.zeros((d,c))  #over written y0
		t = list(range(1,d+1))
		t = np.array(t)
		ind = np.array(ind)
		for out1 in range(0,len(ind)):
				ind[out1] = ind[out1] * d

		update = np.add(t,ind)
		coun = 1
		for col in range(len(y0[0])):
			for row in range(len(y0)):
				if(coun in update):
					y0[row][col]=1
				coun += 1

	# 	#update lamda and mu according to ALM

		muYZ = np.subtract(y0,Z)
		for out1 in range(0,len(muYZ)):
			for out2 in range(0,len(muYZ[0])):
				muYZ[out1][out2] *= mu

		lamda = np.add(lamda,muYZ)
		k = min(mu*rho,10**5)
		mu = k


		# creating objective
		traceE = np.trace(np.dot(E.T,E))
		traceW = np.trace(np.dot(w.T,w))
		traceW *= gamma

		traceY = np.trace(np.dot((np.dot(y0.T,ones(d,d))),y0))
		traceY*= lam

		obj = traceE + traceW + traceY
		objective.append(obj)

	index = np.argmax(y0, axis=1)  # find index of max value row-wise

	return index,y0,objective

def check(Data_ori,gt,index):
	# print gt,index
	n = len(gt)
	uY = list(np.unique(gt))
	nClass = len(uY)
	Y0 = np.zeros((n,1));

	if nClass != max(gt):
		for i in range(nClass):
			for j in range(len(gt)):
				if gt[j] == uY[i]:
					Y0[j]=i

		gt = Y0

	uY = list(np.unique(index))
	nclass = len(uY)
	predY0 = np.zeros((n,1))
	# print uY,nclass,predY0

	if nClass != max(index):
		for i in range(nclass):
			for j in range(len(index)):
				if index[j] == uY[i]:
					index[j]=i

		index = predY0

	Lidx = list(np.unique(gt))
	classnum = len(Lidx)
	predLidx = list(np.unique(index))
	pred_classnum = len(predLidx)

	correnum = 0
	oriDatax = []
	oriDatay = []
	Datax = []
	Datay = []
	for i in range(pred_classnum):
		s = []
		incluster = []

		for j in range(len(index)):
			if index[j] == predLidx[i]:
				s.append(j)

		for j in s:
			incluster.append(gt[j])
			oriDatax.append(max(Data_ori[j]))
			oriDatay.append(gt[j])
			Datax.append(max(Data_ori[j]))
			Datay.append(index[j])
		
		maxIncluster = max(incluster)
		h = list(range(1,maxIncluster+1))
		inclunub = np.histogram(incluster,h)
		inclu = inclunub[0]
		if len(inclu) == 0:
			inclu = [0]
		correnum = correnum*1.0 + max(inclu);
	print index.shape
	mp.scatter(oriDatax,oriDatay,c="blue",label="label1",s=100)	
	mp.scatter(Datax,Datay,c="green",label="label2",s=100)	
	
	mp.show()
	purity = correnum/len(index)
	return purity







#----------------------------------------------------------------------
if __name__ == "__main__":

	data = 'ionosphere.csv' #data file

	print "Initializing data..."
	Data_ori, gt, c = readCSV(data) 
	G = []
	# print Data_ori[43],Data_ori[51]
	for i in range(len(gt)):
		if gt[i]=='g':
			G.append(1)
		else:
			G.append(2)
	gt = G
	Data_ori = np.array(Data_ori)
	clusters = np.random.choice([x for x in xrange(1,c+1,1)],len(Data_ori[0])*1)
	
	diffClass = np.sort(np.unique(clusters))

	y0 = np.zeros((len(Data_ori[0]),len(diffClass)))
	for i in range(len(Data_ori[0])):
		y0[i][np.argwhere(diffClass == clusters[i])] = 1

	index,y,objective =bclsALM(Data_ori,y0,mu)
	index = list(index)

	purity = check(Data_ori,gt,index)
	print "purity: ",purity