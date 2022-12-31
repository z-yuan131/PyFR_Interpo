# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType

from interpolation import Interpolationcls

def main():

    solnname = '/Users/yuanzhenyang/develop/interpolation_use_files/solution_old.pyfrs'
    omeshname = '/Users/yuanzhenyang/develop/interpolation_use_files/mesh_old.pyfrm'

    nmeshname = '/Users/yuanzhenyang/develop/interpolation_use_files/mesh_new.pyfrm'
    nsolnname = '/Users/yuanzhenyang/develop/interpolation_use_files/soln_new.pyfrs'


    sys.argv = [omeshname,solnname,nmeshname,nsolnname]

    Interpolationcls(sys.argv).mainproc()


if __name__ == "__main__":
    main()
