import numpy as np
from datasets import load_dataset

data = load_dataset("riddle_sense")

"""**Task 1:**
Create a new dataset where each possible answer appears only once.
"""

listOfIndicesToSelect = list()
listOfIndicesToRemove = list()
allWordSet = set()

for index, sample in enumerate(data["train"]):
    shouldAddFlag = 1
    for word in sample["choices"]["text"]:
        if word in allWordSet:
            shouldAddFlag = 0
            break
        else:
            allWordSet.add(word)
    if shouldAddFlag:
        listOfIndicesToSelect.append(index)
    else:
        listOfIndicesToRemove.append(index)

newDataset = data["train"].select(listOfIndicesToSelect)


# Test that each possible answer appears only once. Result should be nothing printed

mapDict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
result = []
for sample in newDataset:
    result.append(sample["choices"]["text"][mapDict[sample["answerKey"]]])

result = np.array(result)
a = np.unique(result, return_counts=True)

for index, word in enumerate(a[0]):
    if a[1][index] != 1:
        print(f"word: {word}, times: {a[1][index]}")

# Test that each possible answer appears only once. Result should be all 1's

check = list()
for sample in newDataset:
    for word in sample["choices"]:
        check.append(word)

check = np.array(check)
a = np.unique(result, return_counts=True)
print(a[1])

"""Task 2: Create the following statistic about each possible answer:
Number of times that the word/phrase appeard in a correct answer out of the number of times the word/phrase appeared as a possible answer

"""

mapDict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
result = []
for sample in data["train"]:
    result.append(sample["choices"]["text"][mapDict[sample["answerKey"]]])

result = np.array(result)
a = np.unique(result, return_counts=True)

probs = {}
for index, word in enumerate(a[0]):
    probs[word] = [a[1][index], 0]

for sample in data["train"]:
    for word in sample["choices"]["text"]:
        if word not in probs:
            probs[word] = [0, 1]
        else:
            probs[word][1] += 1

always1 = dict()
always0 = dict()
others = dict()

realProbsDict = {}
for word in probs:
    prob = probs[word][0] / probs[word][1]
    realProbsDict[word] = prob
    if prob == 1:
        always1[word] = 1
    elif prob == 0:
        always0[word] = 0
    else:
        others[word] = prob

"""Number of unique words which appeared in answers"""

x = len(always1) + len(always0) + len(others)
x

"""Of which 27% have only appeared in golden label answers"""

len(always1) / x

"""70% only appeared in incorrect"""

len(always0) / x

"""And only 2% appeared both in golden label answers and incorrect answers."""

len(others) / x
