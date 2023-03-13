import glob
import os
import pandas as pd


work_dir = os.path.abspath(os.path.dirname(__file__))

def main():
    dirs = sorted(glob.glob(work_dir + "/*.csv"))
    for path in dirs:
        df = pd.read_csv(path)
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        df1 = df.iloc[:df.shape[0]-400]
        df2 = df.iloc[df.shape[0]-400:]

        print(dataset_name)
        dataset_dir = work_dir + "/" + dataset_name + "/"
        create_dir(dataset_dir)

        df1.to_csv(dataset_dir + "/" + dataset_name + '_trn.csv',
                   index=False, sep=',')
        df2.to_csv(dataset_dir + "/" + dataset_name + '_tst.csv',
                   index=False, sep=',')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    main()


