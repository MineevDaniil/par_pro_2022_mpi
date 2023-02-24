// Copyright 2022 Mineev Daniil
#include "../../../modules/task_2/mineev_d_gather/gather.h"
#include <cstring>

int gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
           void *recvbuf, int recvcount, MPI_Datatype recvtype,
           int root, MPI_Comm comm) {
    // Проверка на неккоректные данные номера процесса-отправителя и размеры данных
    if (root < 0 || sendcount <= 0 || recvcount <= 0)
        return MPI_ERR_COUNT;
    // Проверка на размер отправляемых и принимаемых буферов
    int sendtype_size, recvtype_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);
    int send_size = sendcount * sendtype_size;
    int recv_size = recvcount * recvtype_size;
    if (send_size != recv_size) return MPI_ERR_COUNT;
    // Получаем данные номера процесса (root) и количество процессов
    int proc_rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    // Если процесс - получатель
    if (proc_rank == root) {
        // Копируем данные себе
        memcpy(reinterpret_cast<int8_t*>(recvbuf) +
               root * recv_size, sendbuf, send_size);
        // Принимаем данные от остальных
        for (int i = 0; i < proc_count; i++) {
            if (i == root) continue;
            MPI_Recv(reinterpret_cast<int8_t*>(recvbuf) + i * recv_size,
                     recvcount, recvtype, i, 0, comm, MPI_STATUS_IGNORE);
        }
    } else  // Отправляем данные root
        MPI_Send(sendbuf, sendcount, sendtype, root, 0, comm);
    // Всё прошло успешно
    return MPI_SUCCESS;
}
