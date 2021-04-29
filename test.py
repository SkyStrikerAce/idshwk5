from sklearn.ensemble import RandomForestClassifier
import numpy as mp

domainlist = []
testlist = []

class Domain:
    def __init__(self, _name, _label, _length, _numcount):
        self.name = _name
        self.label = _label
        self.length = _length
        self.numcount = _numcount

    def returnData(self):
        return [self.length, self.numcount]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def CountNumbers(str):
    count = 0
    for i in str:
        if i.isdigit():
            count = count + 1
    return count

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(',')
            name = tokens[0]
            label = tokens[1]
            length = len(tokens[0])
            numcount = CountNumbers(tokens[0])
            domainlist.append(Domain(name, label, length, numcount))

def initTest(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            name = str(line)
            length = len(line)
            numcount = CountNumbers(line)
            testlist.append(Domain(name, "", length, numcount))

def main():
    initData("train.txt") 
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())  
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    
    initTest("test.txt")
    with open("result.txt", 'w') as f:
        for i in testlist:
            if clf.predict([i.returnData()]) == 1:
                f.write(i.name + ",dga\n")
            else:
                f.write(i.name + ",notdga\n")

if __name__ == '__main__':
    main()