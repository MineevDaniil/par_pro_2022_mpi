// Copyright 2022 Mineev Daniil
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_2/mineev_d_gather/gather.h"
#include <gtest-mpi-listener.hpp>

// Функция для первых 4 тестов
template<typename T>
bool test(int root, MPI_Datatype type) {
    const int SIZE = 5;
    // Получаем данные номера потока (root) и количество потоков
    int proc_rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    // Создаем буферы
    int size = SIZE * proc_count;
    std::vector<T> src(size), dest(size), loc_src(SIZE);
    // Генерируем данные
    if (proc_rank == root)
        for (int i = 0; i < size; src[i++] = i++) {}
    // Отправляем данные
    MPI_Scatter(src.data(), SIZE, type, loc_src.data(), SIZE, type,
                root, MPI_COMM_WORLD);
    // Принимаем данные
    gather(loc_src.data(), SIZE, type, dest.data(), SIZE, type,
           root, MPI_COMM_WORLD);
    // Так как тестируем равенство векторов только для root, то он проверит src == dest
    // Остальные процессы просто будут возвращать true
    return proc_rank != root || src == dest;
}

TEST(MPI_parallel, gather_int_test) {
    // Тест для int
    EXPECT_TRUE(test<int>(0, MPI_INT));
}

TEST(MPI_parallel, gather_float_test) {
    // Тест для float
    EXPECT_TRUE(test<float>(0, MPI_FLOAT));
}

TEST(MPI_parallel, gather_double_test) {
    // Тест для double
    EXPECT_TRUE(test<double>(0, MPI_DOUBLE));
}

TEST(MPI_parallel, gather_root_1_test) {
    // Тест для root == 1
    EXPECT_TRUE(test<int>(1, MPI_INT));
}

TEST(MPI_parallel, gather_int_to_char_test) {
    // Получаем данные номера потока (root) и количество потоков
    int proc_rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    // Создаем буферы
    int size = 5 * proc_count;
    std::vector<int> src(size), dest(size);
    std::vector<char> loc_src(20);
    // Генерируем данные
    if (proc_rank == 0)
        for (int i = 0; i < size; src[i++] = i++) {}
    // Отправляем данные
    MPI_Scatter(src.data(), 5, MPI_INT, loc_src.data(), 20, MPI_CHAR,
                0, MPI_COMM_WORLD);
    // Принимаем данные
    gather(loc_src.data(), 20, MPI_CHAR, dest.data(), 5, MPI_INT,
           0, MPI_COMM_WORLD);
    // Так как тестируем равенство векторов только для root, то он проверит src == dest
    // Остальные процессы просто будут возвращать true
    EXPECT_TRUE(proc_rank || src == dest);
}

TEST(MPI_parallel, gather_error_with_different_sizes) {
    // Получаем данные номера потока (root) и количество потоков
    int proc_rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    // Создаем буферы
    std::vector<int> a(5), b(3);
    // Пытаемся получить данные
    EXPECT_EQ(gather(a.data(), 5, MPI_INT, b.data(), 3, MPI_INT,
                     0, MPI_COMM_WORLD), MPI_ERR_COUNT);
}

TEST(MPI_parallel, gather_negative_root) {
    // Получаем данные номера потока (root) и количество потоков
    int proc_rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    // Создаем буферы
    std::vector<int> a(5), b(5);
    // Пытаемся получить данные
    EXPECT_EQ(gather(a.data(), 5, MPI_INT, b.data(), 5, MPI_INT,
                     -1, MPI_COMM_WORLD), MPI_ERR_COUNT);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());
    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);

    return RUN_ALL_TESTS();
}
