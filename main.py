# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType

from interpo import Interpo

def main():

    oininame = 'a'
    solnname = './files/post/solution2.pyfrs'
    omeshname = './files/post/mesh.pyfrm'

    nmeshname = './files/mesh_new2.pyfrm'


    sys.argv = [oininame,omeshname,solnname,omeshname]

    Interpo(sys.argv).getID()

    #print(oininame)




if __name__ == "__main__":
    main()
