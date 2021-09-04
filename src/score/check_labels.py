import pandas as pd


def main():
    df = pd.read_csv('/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/ml_score/data/names.csv')
    print(df.label.unique())


if __name__=="__main__":
    main()