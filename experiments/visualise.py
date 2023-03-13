import pandas as pd
import matplotlib.pyplot as plt
import os

parent_dir = os.path.abspath(os.path.dirname(__file__))
graph_dir = parent_dir + "/output/graphs"
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)


class StockGraph:
    def __init__(self, dir, name, label):
        self.dir = dir
        self.df = None
        self.tickers = []
        self.datetime = []
        self.name = name
        self.label = label
        self.color = "blue"

    def load_data(self):
        stock_df = pd.read_csv(self.dir)
        stock_df.rename(columns= {"AVG":self.label},inplace=True)
        self.df = stock_df
        self.tickers = stock_df[self.label]
        self.datetime = stock_df["Date"]

    def plot_graph(self,ax):
        self.df.plot(kind='line', x='Date', y=self.label,ax = ax)

    def get_size(self):
        return len(self.tickers)

    def set_color(self,color):
        self.color = color

    def change_size(self, begin, end):
        try:
            self.tickers = self.tickers[begin:end]
            self.datetime = self.datetime[begin:end]
            self.df = self.df[begin:end]
        except ValueError:
            print("Error in changing size of the graph data")

def save_plot(name):
    plt.legend()
    total_path = graph_dir + "/" + name + '.png'
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(total_path, format="png", dpi=300)

def plot_single_stock(graph):
    ax = plt.gca()
    graph.plot_graph(ax)
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.title("The original graph of the price")
    save_plot(graph.name)
    plt.clf()


def plot_with_predictions(graph1,graph2):
    ax = plt.gca()
    graph1.change_size(0,graph2.get_size())
    graph1.plot_graph(ax)
    graph2.plot_graph(ax)
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.title("The original graph of the price")
    save_plot("Original with predictions")
    plt.clf()



def main():
    data_path = "data/Energy/Energy_tst.csv"
    simple_graph = StockGraph(data_path, "original_data",label="ORI_DATA")
    simple_graph.load_data()

    data_path2 = "output/predictions/pred_y_03_11_2023_14_22_38.csv"
    pred_graph = StockGraph(data_path2, "predictions",label="PRED")
    pred_graph.load_data()

    simple_graph.set_color("blue")
    pred_graph.set_color("red")
    plot_with_predictions(graph1=simple_graph,graph2=pred_graph)


if __name__ == '__main__':
    main()
