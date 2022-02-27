import pandas as pd
import numpy as np

class MyTopsis:

    def __init__(self, input_file, weight, impact, out_file):
        self.input_file = input_file
        self.weight = weight
        self.impact = impact
        self.out_file = out_file

    def calculate(self):
        w = self.weight.split(',')
        try:
            w = [int(i) for i in w]
        except ValueError:
            print("Weights should only be numbers\n")
            exit()

        impacts = self.impact.split(',')
        for i in impacts:
            if i != '+' and i != '-':
                print("impacts should be either + or -")
                exit()

        try:
            read_file = pd.read_excel(self.input_file)
            read_file.to_csv('101917196-data.csv', index=None, header=True)
            df = pd.read_csv("101917196-data.csv")
        except FileNotFoundError:
            print("File not found")
            exit()

        if len(df.columns) < 3:
            print("Input file must contain three or more columns.\n")
            exit()

        check = {len(df.columns)-1, len(w), len(impacts)}
        if len(check) != 1:
            print(
                "Number of Weights, number of impacts and number of indicators must be same.\n")
            exit()

        for col in df.iloc[:, 1:]:
            for i in df[col]:
                if isinstance(i, float) == False:
                    print("columns must contain numeric values only\n")
                    exit()

        arr = np.array(df.iloc[:, 1:])

        root_sum_of_squares = np.sqrt(np.sum(arr**2, axis=0))

        arr = np.divide(arr, root_sum_of_squares)
        arr = arr*w

        ideals = np.zeros((arr.shape[1], 2))
        for i in range(len(impacts)):
            l = np.zeros(2)
            if impacts[i] == '+':
                l[0] = max(arr[:, i])
                l[1] = min(arr[:, i])
            elif impacts[i] == '-':
                l[0] = min(arr[:, i])
                l[1] = max(arr[:, i])
            ideals[i, 0] = l[0]
            ideals[i, 1] = l[1]
        ideals = ideals.T

        distances = np.zeros((arr.shape[0], 2))

        for i in range(arr.shape[0]):
            best_dist = np.linalg.norm(arr[i, :] - ideals[0, :])
            worst_dist = np.linalg.norm(arr[i, :] - ideals[1, :])
            distances[i, 0] = best_dist
            distances[i, 1] = worst_dist

        performance_score = np.divide(
            distances[:, 1], np.add(distances[:, 0], distances[:, 1]))

        rank = np.zeros(arr.shape[0])

        temp = list(performance_score)
        count = 1
        for i in range(len(performance_score)):
            ind = np.argmax(temp)
            rank[ind] = count
            count += 1
            temp[ind] = -99

        df_out = df
        df_out['Topsis Score'] = performance_score
        df_out['Rank'] = rank
        df_out.to_csv(self.out_file, index=None)
        print("Completed succesfully! Check " +
              self.out_file+" for the output\n")