import pyspark
import sys
import collections

def apriori(chunk, ps):
    # calculate the first phrase
    pre = []
    singleCount = collections.defaultdict(int)
    chunk = list(chunk)
    for basket in chunk:
        for item in basket:
            if singleCount[item] < ps:
                singleCount[item] += 1
                if singleCount[item] >= ps:
                    pre.append((item,))
                    yield item,
    curLen = 2
    # the loop stops when there's no frequent item sets in this phrase
    while pre:
        candidates = set()
        for i in range(len(pre) - 1):
            iSet = set(pre[i])
            for j in range(i + 1, len(pre)):
                jSet = set(pre[j])
                toAdd = iSet | jSet
                if len(toAdd) == curLen:
                    candidates.add(tuple(sorted(toAdd)))
        curCount = collections.defaultdict(int)
        pre = []
        for basket in chunk:
            basketSet = set(basket)
            for candidate in candidates:
                if set(candidate).issubset(basketSet):
                    if curCount[candidate] >= ps:
                        continue
                    curCount[candidate] += 1
                    if curCount[candidate] >= ps:
                        pre.append(candidate)
                        yield candidate
        curLen += 1

if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder.appName("Son_algorithm_app").config("spark.some.config.option", "some-value").getOrCreate()
    # partionNum and support threshold needs to be specified
    partitionNum = 4
    support = int(sys.argv[2])
    ps = support / partitionNum
    # to get all item sets
    data = spark.read.format("csv").option("header","true").load(sys.argv[1]).rdd.map(lambda line: (int(line[0]), int(line[1]))).groupByKey().map(lambda x: set(x[1])).repartition(partitionNum)
    # get frequent sets calculated by each partition
    partitionRes = data.mapPartitions(lambda chunk: apriori(chunk, ps))
    # filter away negative positive item sets
    frequentSets = []
    countDict = collections.defaultdict(int)
    for item in set(partitionRes.collect()):
        itemSet = set(item)
        for basket in data.collect():
            if not itemSet.issubset(basket):
                continue
            if countDict[item] < support:
                if countDict[item] + 1 >= support:
                    frequentSets.append(item)
            countDict[item] += 1
    frequentSets.sort()
    # confidence need to be specified
    confidence = float(sys.argv[3])
    # find out all single items
    singleItems = set()
    for item in frequentSets:
        if len(item) == 1:
            singleItems.add(item)
    # find association with confidence larger than threshold
    confidentAssociate = []
    count = 0
    allCount = 0
    for I in frequentSets:
        for j in singleItems:
            count += 1
            allCount += 1
            if count == 200:
                print(allCount)
                count = 0
            if j[0] in I:
                continue
            supI = countDict[I]
            IWithJ = tuple(sorted(I + j))
            if IWithJ not in countDict:
                supIWithJ = 0
                IWithJSet = set(IWithJ)
                for basket in data.collect():
                    if not IWithJSet.issubset(basket):
                        continue
                    supIWithJ += 1
                countDict[IWithJ] = supIWithJ
            curConfidence = countDict[IWithJ] / supI
            if curConfidence > confidence:
                confidentAssociate.append((I, j, curConfidence))
    confidentAssociate.sort()
    # output file
    with open("output.txt", 'w') as f:
        f.write("Frequent itemset \n")
        for eachSet in frequentSets:
            f.write(','.join(str(item) for item in eachSet))
            f.write('\n')
        f.write("Confidence \n")
        for eachAssociation in confidentAssociate:
            f.write(str(eachAssociation[0]) + ',' + str(eachAssociation[1][0]) + ' ' + str(eachAssociation[2]))
            f.write('\n')
