import sklearn.datasets as datasets
from sklearn import tree
import pandas as pd
import sys


def usage():
    print("Usage: python your_script.py sepal_length sepal_width petal_length petal_width")
    print("Example: python your_script.py 5.1 3.5 1.4 0.2")
    print("Note: Provide the sepal length, sepal width, petal length, and petal width as command-line arguments.")


data = datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

y = data.target
dtree = tree.DecisionTreeClassifier()
dtree.fit(df, y)

if len(sys.argv) != 5:
    usage()
else :
    try:
        sepal_length = float(sys.argv[1])
        sepal_width = float(sys.argv[2])
        petal_length = float(sys.argv[3])
        petal_width = float(sys.argv[4])


        result = dtree.predict([[
            sepal_length, sepal_width, 
            petal_length, petal_width
            ]])

        if result == 0:
            print("The setosa")
        elif result == 1:
            print("The versicolor")
        elif result == 2:
            print("The virginica")
        else:
            print("I don't know")

    except:
        print("Please enter the correct values")

