import math as m

def parse_file(filename):
  propBlack = 0
  topProp = 0
  leftProp = 0
  leftBlack = 0
  topBlack = 0
  totalBlack = 0
  inputMap = []
  with open(filename) as rawInput:
    for i in rawInput:
      l = i.strip()
      inputMap.append(l.split(','))
  yCounter = 0
  xCounter = 0
  xLength = m.floor(len(inputMap[0])/2)
  yLength = m.floor(len(inputMap)/2)
  for i in inputMap:
    for x in i:
      if (x == '1'):
        totalBlack += 1
        if (xCounter < xLength):
          leftBlack += 1
        if (yCounter < yLength):
          topBlack += 1
      xCounter += 1
    yCounter += 1
    xCounter = 0
  propBlack = totalBlack/(len(inputMap[0]) * len(inputMap))
  topProp = topBlack/totalBlack
  leftProp = leftBlack/totalBlack
  return propBlack, topProp, leftProp
def probLetter(prop, mu, sig):
  total = (1/(m.sqrt(2 * m.pi * (sig ** 2)))) * \
      (m.e ** (-1/2 * (((prop - mu)/sig) ** 2)))
  return total 
def normalize(arr):
  return [x/sum(arr) for x in arr]

def naive_bayes_classifier(input_filepath):
  # input is the full file path to a CSV file containing a matrix representation of a black-and-white image
  letterDict = {'A':[0.38, 0.06, 0.46, 0.12, 0.50, 0.09], 'B':[0.51, 0.06, 0.49, 0.12, 0.57, 0.09],
  'C':[0.31, 0.06, 0.37, 0.09, 0.64, 0.06], 'D':[0.39, 0.06, 0.47, 0.09, 0.57, 0.03], 
  'E':[0.43, 0.12, 0.45, 0.15, 0.65, 0.09]}
  propBlackArr = []
  topPropArr = []
  leftPropArr = []
  propBlack, topProp, leftProp = parse_file(input_filepath)
  for i in letterDict:
    propBlackArr.append(probLetter(propBlack, letterDict[i][0], letterDict[i][1]))
    topPropArr.append(probLetter(topProp, letterDict[i][2], letterDict[i][3]))
    leftPropArr.append(probLetter(leftProp, letterDict[i][4], letterDict[i][5]))
  finalArr = []
  priorList = [0.28, 0.05, 0.10, 0.15, 0.42]
  nums = [0,0,0]
  for i in range(5):
    nums[0] += priorList[i] * propBlackArr[i]
    nums[1] += priorList[i] * topPropArr[i]
    nums[2] += priorList[i] * leftPropArr[i]
  denominator = nums[0] * nums[1] * nums[2]
  for i in range(len(propBlackArr)):
    finalArr.append((priorList[i] * propBlackArr[i] * topPropArr[i] * leftPropArr[i])/denominator)
  class_probabilities = normalize(finalArr)
  letterList = ['A', 'B', 'C', 'D', 'E']
  most_likely_class = letterList[class_probabilities.index(max(class_probabilities))]

  return most_likely_class, class_probabilities

  # most_likely_class is a string indicating the most likely class, either "A", "B", "C", "D", or "E"
  # class_probabilities is a five element list indicating the probability of each class in the order [A probability, B probability, C probability, D probability, E probability]
  #return most_likely_class, class_probabilities

def fuzzyLogic(a, b, c, d, x):
  if (x <= a):
    return 0
  elif (a < x and x < b):
    return ((x-a)/(b-a))
  elif (b <= x and x <= c):
    return 1
  elif (c < x and x < d):
    return ((d - x)/(d - c))
  elif (d <= x):
    return 0

  return 0

def fuzzyAnd(arg1, arg2):
  return min(arg1, arg2)

def fuzzyOr(arg1, arg2):
  return max(arg1, arg2)

def fuzzy_classifier(input_filepath):
  # input is the full file path to a CSV file containing a matrix representation of a black-and-white image
  propBlackFuzzyDict = {'Low':[0, 0, 0.3, 0.4], 'Medium':[0.3, 0.4, 0.4, 0.5], 'High':[0.4, 0.5, 1, 1]}
  topPropFuzzyDict = {'Low':[0, 0, 0.3, 0.4], 'Medium':[0.3, 0.4, 0.5, 0.6], 'High':[0.5, 0.6, 1, 1]}
  leftPropFuzzyDict = {'Low':[0, 0, 0.3, 0.4], 'Medium':[0.3, 0.4, 0.6, 0.7], 'High':[0.6, 0.7, 1, 1]}
  helperArr = ['Low', 'Medium', 'High']
  letterList = ['A', 'B', 'C', 'D']
  propBlackValues = []
  topPropValues = []
  leftPropValues = []
  propBlack, topProp, leftProp = parse_file(input_filepath)

  for i in helperArr:
    propBlackValues.append(fuzzyLogic(
        propBlackFuzzyDict[i][0], propBlackFuzzyDict[i][1], propBlackFuzzyDict[i][2], propBlackFuzzyDict[i][3], propBlack))
    topPropValues.append(fuzzyLogic(
        topPropFuzzyDict[i][0], topPropFuzzyDict[i][1], topPropFuzzyDict[i][2], topPropFuzzyDict[i][3], topProp))
    leftPropValues.append(fuzzyLogic(
        leftPropFuzzyDict[i][0], leftPropFuzzyDict[i][1], leftPropFuzzyDict[i][2], leftPropFuzzyDict[i][3], leftProp))
  class_memberships = []
  #if PropBlack is Medium and (TopProp is Medium or LeftProp is Medium) then class A
  class_memberships.append(fuzzyAnd(propBlackValues[1], fuzzyOr(topPropValues[1], leftPropValues[1])))
  #IF PropBlack is High AND TopProp is Medium AND LeftProp is Medium THEN class B.
  class_memberships.append(fuzzyAnd(propBlackValues[2], fuzzyAnd(topPropValues[1], leftPropValues[1])))
  #IF(PropBlack is Low AND TopProp is Medium) OR LeftProp is High THEN class C.
  class_memberships.append(fuzzyOr(fuzzyAnd(propBlackValues[0], topPropValues[1]), leftPropValues[2]))
  #IF PropBlack is Medium AND TopProp is Medium AND LeftProp is High THEN class D.
  class_memberships.append(fuzzyAnd(propBlackValues[1], fuzzyAnd(topPropValues[1], leftPropValues[2])))
  #IF PropBlack is High AND TopProp is Medium AND LeftProp is High THEN class E.
  class_memberships.append(fuzzyAnd(propBlackValues[2], fuzzyAnd(topPropValues[1], leftPropValues[2])))
  if(len(class_memberships) > 0):
    highest_membership_class = letterList[class_memberships.index(max(class_memberships))]
    return highest_membership_class, class_memberships
  else:
    print("NO LETTER FOUND")
    return 0
