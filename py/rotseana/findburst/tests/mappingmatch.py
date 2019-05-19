'''
Created on Nov 5, 2017

@author: daniel
'''

import rotseana.findburst.tests.mappingcommon as mc
from scipy.io import readsav


# function that gets match and prints elements structure of that file.

INDENT = '|--'


def elements(file):
    mfile = readsav(file, python_dict=True)
    etree = mc.elementtree(mfile)
    return etree


if __name__ == '__main__':
    import os
    heredir = os.path.dirname(os.path.abspath(__file__))
    basedir = os.path.dirname(heredir)
    datdir = os.path.join(basedir, 'dat')
    filename = '000409_xtetrans_1a_match.dat'
    datfile = os.path.join(datdir, filename)
    mdata = elements(datfile)
    print(repr(mdata))

    # print(mdata.repr_flat())
