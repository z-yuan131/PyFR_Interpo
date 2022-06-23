# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType

from interpo import Interpo

def main():

    oininame = 'a'
    solnname = './files/solution.pyfrs'
    omeshname = './files/mesh.pyfrm'
    nmeshname = 'b'


    sys.argv = [oininame,omeshname,solnname,omeshname]

    Interpo(sys.argv).getID()

    #print(oininame)




if __name__ == "__main__":
    main()
