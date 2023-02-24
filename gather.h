// Copyright 2022 Mineev Daniil
#ifndef MODULES_TASK_2_MINEEV_D_GATHER_GATHER_H_
#define MODULES_TASK_2_MINEEV_D_GATHER_GATHER_H_
#include <mpi.h>

int gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
           void *recvbuf, int recvcount, MPI_Datatype recvtype,
           int root, MPI_Comm comm);

#endif  // MODULES_TASK_2_MINEEV_D_GATHER_GATHER_H_
