# Задание 1, вариант 3

Решение уравнения Пуассона в трапеции: последовательная версия, OpenMP, MPI и MPI+OpenMP.

[Отчёт](report/report.pdf) · [Исходник LaTeX](report/report.tex)

## Сборка на Полюсе

Из корня репозитория:

```bash
cd task1
module load SpectrumMPI
make all
```

## Запуск

```bash
bash submit.sh seq
bash submit.sh omp
bash submit.sh mpi
bash submit.sh hybrid
```

Каждая команда отправляет одно задание в очередь. Сетки и число процессов и потоков взяты из приложения к заданию.

Результаты лежат в `results/`, время и число итераций — в файлах `*_summary.txt`. Очередь можно посмотреть командой `bjobs`.

При новом подключении к Полюсу нужно снова выполнить `module load SpectrumMPI`.
