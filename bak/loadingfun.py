#from pyfr.shapes import BaseShape
#from pyfr.util import lazyprop, subclass_where
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
#from collections import defaultdict
#import numpy as np
#from tqdm import tqdm


def load_ini_info(ini_name):
    from argparse import ArgumentParser, FileType

    ap = ArgumentParser(description='Read interpolation argument.')
    ap.add_argument('--data_dir', type=str, default='./channel_Retau180.ini',
                       help='data directory containing input.txt')
    ap.add_argument('cfg', type=FileType('r'), help='config file')

    args = ap.parse_args([ini_name])
    confg = Inifile.load(args.cfg)

    return confg

def load_soln_info(soln_name):
    from argparse import ArgumentParser, FileType

    ap = ArgumentParser(description='Read interpolation argument.')
    ap.add_argument('--data_dir', type=str, default='./100.pyfrs',
                       help='data directory containing input.txt')
    ap.add_argument('solnf', type=FileType('r'), help='solution file')

    args = ap.parse_args([soln_name])
    #print(args)
    soln = NativeReader(args.solnf.name)

    return soln

def load_mesh_info(mesh_name):
    from argparse import ArgumentParser, FileType

    ap = ArgumentParser(description='Read interpolation argument.')
    ap.add_argument('--data_dir', type=str, default='./100.pyfrs',
                       help='data directory containing input.txt')
    ap.add_argument('meshf', type=FileType('r'), help='solution file')

    args = ap.parse_args([mesh_name])
    #print(args.meshf.name)
    mesh = NativeReader(args.meshf.name)

    return mesh


#cfg = load_ini()   #'/cylinder/cylinder.ini'
#Mysoln = load_soln()   #'/cylinder/cylinOrd5_20.00.pyfrs'
#Mymesh_old = load_mesh('./channel/mesh_channel.pyfrm')
#Mymesh_new = load_mesh('mesh_new.pyfrm')
