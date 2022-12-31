# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType

from interpo import Interpo

def main():

    oininame = 'a'
    solnname = './files/post/solution2.pyfrs'
    omeshname = './files/post/mesh.pyfrm'

    # more partitions
    solnname = './files/solution2.pyfrs'
    omeshname = './files/mesh.pyfrm'

    nmeshname = './files/mesh_new2.pyfrm'

    # channel case 3D
    solnname = './large_files/channel_1000.00.pyfrs'
    omeshname = './large_files/mesh_channel.pyfrm'

    nmeshname = './large_files/refine.pyfrm'


    sys.argv = [oininame,omeshname,solnname,nmeshname]

    Interpo(sys.argv).getID()

    #print(oininame)




if __name__ == "__main__":
    main()
