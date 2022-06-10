import argparse
import pickle
import matplotlib.pyplot as plt

def main(filename):
    fig = pickle.load(open(f'{filename}','rb'))
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("picklefile", type = str, help = 'Path to pickled light curve plot')
args = parser.parse_args()
filename = args.picklefile
main(filename)
