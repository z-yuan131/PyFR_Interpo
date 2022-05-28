from mpi4py import MPI
import sys

from search import hide_och_catch,hide


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def main():
    # import the newmesh name and oldmesh name and oldsolution name from command line
    if len(sys.argv) != 4:
        raise RuntimeError('Three positional arguments are needed: Oldmesh, Oldsoln, Newmesh')

    ininame = '../channel/channel_Retau180.ini'
    solnname = '../channel/100.pyfrs'
    omeshname = '../channel/mesh_channel.pyfrm'
    nmeshname = '../channel/mesh_new_less_less.pyfrm'

    sys.argv = [ininame,omeshname,solnname,nmeshname]



    if rank == 0:
        if size > 1:
            # firstly check if partition is the same as the mpi rank
            if size == hide_och_catch(sys.argv).getpartition() + 1:
                # secondly partition the new mesh into the same number of partitions
                # it is worthwhile to check if pyfr partition also does the same
                #print('checking partition of the new mesh is the same size of the old one')
                hide_och_catch(sys.argv).ptN(size)


                hide(sys.argv).getID(rank, size, comm)


            else:
                raise RuntimeError('One can only run this code with same partition number as the mesh or zero partition')
        else:
            # do collect all elements into one piece
            print('old algorithm')

    else:
        hide(sys.argv).getID(rank,size,comm)




if __name__ == "__main__":
    main()
