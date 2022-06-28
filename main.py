# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType

from interpo import Interpo

def main():

    oininame = 'a'
    solnname = './files/solution.pyfrs'
    omeshname = './files/mesh.pyfrm'
    nmeshname = './files/mesh_new.pyfrm'


    sys.argv = [oininame,omeshname,solnname,nmeshname]

    Interpo(sys.argv).getID()

    #print(oininame)




if __name__ == "__main__":
    main()
