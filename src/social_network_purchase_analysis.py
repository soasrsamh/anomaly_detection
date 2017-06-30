##################################################################################################################
##################################################################################################################
###### Social Network Purchase Analysis
###### S.L. Howell June 2017
###### Insight Data Engineering Coding Challenge 
######
###### Analyzes purchases within a social network of users and detects any behavior 
###### that is far from the average within that social network.
##################################################################################################################
##################################################################################################################




##################################################################################################################
#### Imports
##################################################################################################################
import json
import sys
from math import sqrt
from math import floor
from operator import itemgetter
##################################################################################################################




##################################################################################################################
#### Input and Output files
##################################################################################################################
# batch_log.json contains past data that is used to build the initial state 
# of the entire user network and purchase history.
initialfilepath = sys.argv[1]
#initialfilepath = "C:/Users/Sarah/Desktop/anomaly_detection-master/sample_dataset/batch_log.json" 

# stream_log.json contains new purchases including possible anomalous purchases.
streamfilepath  = sys.argv[2]
#streamfilepath = "C:/Users/Sarah/Desktop/anomaly_detection-master/sample_dataset/stream_log.json"

# If a purchase is flagged as anomalous, it is logged in flagged_purchases.json. 
flaggedfilepath = sys.argv[3]
#streamfilepath = "C:/Users/Sarah/Desktop/flagged_purchases.json"
##################################################################################################################




##################################################################################################################
#### Classes
##################################################################################################################
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
##################################################################################################################
class Vertex:
	def __init__(self,key,T):
		self.id = key
		self.connectedTo = {}
		self.color = 'white'
		self.dist = sys.maxsize
		self.pred = None
		self.T =T
		#A deque with a max length was used so that most unnecessary historical purchases are not stored.
		self.purchases = []
		
	def addNeighbor(self,nbr,weight=0):
		self.connectedTo[nbr] = weight
		
	def delNeighbor(self,nbr):
		del self.connectedTo[nbr]
		
	def setColor(self,color):
		self.color = color
		
	def setDistance(self,d):
		self.dist = d
	
	def setPred(self,p):
		self.pred = p
		
	def getPred(self):
		return self.pred
	
	def addPurchase(self, purchasenumber, amount):
		self.purchases.append([purchasenumber, amount])
		self.purchases = self.purchases[-T:]
		
	def getPurchases(self):
		return self.purchases
	
	def getDistance(self):
		return self.dist
	
	def getColor(self):
		return self.color
	
	def __str__(self):
		return str(self.id) + ":color " + self.color + ' connectedTo: ' + str([x.id for x in self.connectedTo])
       
	def getConnections(self):
		return self.connectedTo.keys()

	def getId(self):
		return self.id

	def getWeight(self,nbr):
		return self.connectedTo[nbr]
##################################################################################################################
class Graph:
    def __init__(self,T):
        self.vertList = {}
        self.numVertices = 0
        self.T = T

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key,T)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)
        
    def delEdge(self,f,t):
        self.vertList[f].delNeighbor(self.vertList[t])
		
    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())
##################################################################################################################




##################################################################################################################
#### Functions
##################################################################################################################
# A purchase amount is anomalous if it's greater than 3 standard deviations 
# above the mean of the last T purchases in the user's Dth degree social network.
def record_if_anomalous(amount, network, start, T, D):
	numberofpurchases, mean, sd = bfs(network,network.getVertex(start),T,D)
	if numberofpurchases > 1:
		if amount > mean + 3*sd: #Then it is anomalous, so record it with strings to two decimal places trunicated down.
			return dict(event, **{"mean":"{0:.2f}".format(floor(mean*100)/100),"sd":"{0:.2f}".format(floor(sd*100)/100)})
	return False

def add_purchase(friendnetwork, event, purchasenumber):
	friend = int(event["id"])
	if friend not in friendnetwork:
		friendnetwork.addVertex(friend)
	friendnetwork.vertList[friend].addPurchase(purchasenumber, float(event["amount"]))
	
def add_friend(friendnetwork, event):
	friend1= int(event["id1"])
	friend2 = int(event["id2"])
	friendnetwork.addEdge(friend1,  friend2)
	friendnetwork.addEdge(friend2,  friend1)
			
def del_friend(friendnetwork, event):
	friendnetwork.delEdge(int(event["id1"]),  int(event["id2"]))
	friendnetwork.delEdge(int(event["id2"]),  int(event["id1"]))
	
# A breadth first search (BFS) is given a graph, g, and a starting vertex, start, and the max number of degrees, D.
# It proceeds by exploring edges in the graph to find the all vertices in g for which there is a path from s
# that are a distance D or less from the starting vertex.
def bfs(g,start,T,D):
	purchaselist = []
	vertlist = Vertex("delete",T)
	# BFS begins at the starting vertex s and colors start gray to show that it is currently being explored.
	# The distance and the predecessor are initialized to 0 and None respectively for the starting vertex.
	start.setDistance(0) 
	start.setPred(None)
	#start is placed on a new Queue.
	vertQueue = Queue()
	vertQueue.enqueue(start)
	# Each new node at the front of the queue is systematically explored.
	while (vertQueue.size() > 0):
		currentVert = vertQueue.dequeue()
		#It stops searching before it finds any vertices that are a distance greater than D away.
		if currentVert.getDistance()==D:
			break
		#Each new node on the adjacency list is iterated over.
		for nbr in currentVert.getConnections():
			#As each node on the adjacency list is examined its color is checked.
			#If its color is white (this means the vertex is unexplored), then explore it.
			if (nbr.getColor() == 'white'): 
				#make a list of all the friends' last T purchases
				purchaselist = (sorted(purchaselist+nbr.getPurchases(), key=itemgetter(0)))[-T:]
				# Tried to use linear_merge (code is commented below) instead of sorted because the two lists are already sorted, 
				# but it was slower for some reason on the sample data.
				#purchaselist = linear_merge(purchaselist,nbr.getPurchases(),T) 
				#print("purchaselist of", start.id, "with id", nbr.id, "added is", purchaselist)
				vertlist.addNeighbor(nbr)
				# The new, unexplored vertex nbr, is colored gray indicating that it is being explored and 
				# may have some white vertices adjacent to it.
				nbr.setColor('gray') 
				# The distance to nbr is set to the distance to currentVert + 1
				nbr.setDistance(currentVert.getDistance() + 1)
				# The predecessor of nbr is set to the current node currentVert
				nbr.setPred(currentVert)
				# nbr is added to the end of a queue. Adding nbr to the end of the queue effectively schedules 
				# this node for further exploration, but not until all the other vertices on the adjacency list 
				# of currentVert have been explored.
				vertQueue.enqueue(nbr)
		# When BFS has completely explored a vertex it is colored black, 
		# meaning it has no white vertices adjacent to it.
		currentVert.setColor('black')
		vertlist.addNeighbor(currentVert)
	# Reset the touched vertices to white
	for neighbor in vertlist.getConnections():
		neighbor.setColor('white')
	
	# Calculate and return the numberofpurchases, mean, and the sd
	purchaselist=[row[1] for row in purchaselist]  
	numberofpurchases = len(purchaselist)
	if (numberofpurchases > 1):
		mean = sum(purchaselist)/numberofpurchases
		sd = 0.0
		for purchase in purchaselist:
			sd += (purchase-mean)**2
		return numberofpurchases, mean, sqrt(sd/numberofpurchases)
	return numberofpurchases, 0, 0
	
#returns the sorted last T items from two sorted purchase lists
def linear_merge(list1, list2,T):
    merged_list = []
    i = 0
    j = 0
    try:
        while True:
            if list1[i][0] <= list2[j][0]:
                merged_list.append(list1[i])
                i += 1
            else:
                merged_list.append(list2[j])
                j += 1
    except IndexError:
        if i == len(list1):
            merged_list.extend(list2[j:])
        if j == len(list2):
            merged_list.extend(list1[i:])
    return merged_list[-T:]
##################################################################################################################




##################################################################################################################
##### Flexible Parameters
##### The first line of batch_log.json contains a JSON object with the parameters: 
##### Degree (D) and number of tracked purchases (T) to consider for the calculation.
##################################################################################################################
# Open file, batch_log.json, containing the flexible parameters on the first line and read the line.
with open(initialfilepath, 'r') as initialfile:
	firstline = eval(initialfile.readline()) #read line as a dict
	print (firstline) 

# The number of degrees (D) that defines a user's social network will be greater than 1. 
# A value of 1 means only the friends of the user are considered.
# A value of 2 means the social network extends to friends and "friends of friends".
	D = int(firstline["D"])

# The tracked number of purchases in the user's network (T) will be at least 2 and is
# the number of consecutive purchases made by a user's social network (not including the user's own purchases).
	T = int(firstline["T"])
##################################################################################################################
 
 

##################################################################################################################
##### Input initial state data
##################################################################################################################
# batch_log.json, which is already open, contains past data that is used to build the initial state 
# of the entire user network and purchase history.

# A Graph abstract data type with undirected, unweighted edges is used with an adjacency list implementation
# to store the assumed sparsely connected data.
	friendnetwork = Graph(T)
	
#Add events to build the initial friend network.
	purchasenumber = 1
	for line in initialfile: #for each event line
		try:
			event = eval(line) #convert the line into a dict
		except SyntaxError as err:
			pass #Unexpected input, do nothing
		if event["event_type"] == "purchase":
			add_purchase(friendnetwork, event, purchasenumber)
			#print("Purchase number", purchasenumber, "by", event["id"], "of", event["amount"])
			purchasenumber += 1
		elif event["event_type"] == "befriend":
			add_friend(friendnetwork, event)
			#print("Befriend", event["id1"], "and", event["id2"])
		elif event["event_type"] == "unfriend":
			del_friend(friendnetwork, event)
			#print("Unfriend", event["id1"], "and", event["id2"])
##################################################################################################################



	
##################################################################################################################
##### Stream input data and record flagged purchases
##################################################################################################################
# streamfilepath contains new purchases including possible anomalous purchases.
# If a purchase is flagged as anomalous, it is logged in the flaggedfilepath. 
# As events come in, both the social network and the purchase history of users are updated.
with open(streamfilepath, 'r') as streamfile, open(flaggedfilepath, 'w') as flaggedfile:
	for line in streamfile:
		try:
			event = eval(line) #convert line into a dict
		except SyntaxError as err:
			pass #Unexpected input, do nothing
		if event["event_type"] == "purchase":
			add_purchase(friendnetwork, event, purchasenumber)
			#print("Purchase number", purchasenumber, "by", event["id"], "of", event["amount"])
			record = record_if_anomalous(float(event["amount"]), friendnetwork, int(event["id"]), T, D)
			if record != False: # if the record is anomalous, then add the event to the flagged file.
				print(record)
				flaggedfile.write(str(record) + '\n')
			purchasenumber += 1
		elif event["event_type"] == "befriend":
			add_friend(friendnetwork, event)
			#print("Befriend", event["id1"], "and", event["id2"])
		elif event["event_type"] == "unfriend":
			del_friend(friendnetwork, event)
			#print("Unfriend", event["id1"], "and", event["id2"])
##################################################################################################################
