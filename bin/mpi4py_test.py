from mpi4py import MPI
import numpy as np
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = {'key1' : [7, 2.72, 2+3j],
            'key2' : ( 'abc', 'xyz')}
else:
    data = {'key1' : [1],
            'key2' : ( 'abc', 'xyz')}

#print(rank)
"""
data = comm.bcast(data, root=0)

if rank == 1:
    print(data)



for i in range(size):
    data1 = comm.bcast(data, root=i)
    if rank == 0:
        data = 2
    if rank == 1:
        print(rank,data1)
    #comm.Barrier()
"""


sendbuf = np.zeros(10, dtype='i') + rank
recvbuf = None

#print(sendbuf)
comm.Barrier()
#if rank == 0:
#    recvbuf = np.empty([size, 10], dtype='i')










def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))



A = nested_dict(1, float)


A[0] = list((rank,'I like p0'))
A[1] = list((rank,'I like p1'))
A[2] = list((rank,'I like p2'))
A[3] = list((rank,'I like p3'))

rev = nested_dict(1, float)
for i in range(size):
    rev = comm.gather(A[i], root=i)
    if rank == i:
        #print(rank,rev)

        st = rev
#comm.Barrier()


for i in range(size):
    if i == rank:
        print(len(st))

    #break
#if rank == 1:
#    print(rank,st)
"""
if rank == 0:
    comm.send(A, dest = 1, tag = 1)
if rank == 1:
    A.update(comm.recv(source = 0, tag = 1))
"""

#for i in range(size):
#    print(recvbuf)
#    assert np.allclose(recvbuf[i,:], i)
#print(rev,rank)
