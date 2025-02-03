import pandas as pd
from pyscript import display
from js import console
from pyweb import pydom


class marks_vs_rank:
    def __init__(self):
        pass

    def fetch_data(self):
        pass

def predict(event):
    df = pd.read_csv("C:\\Users\Gudic\IdeaProjects\JEEM_Prediction\Data\\2024\\NITs_2024.csv")
    print(df)
    display(df, target = "result", append = False)
