# Задание 1, вариант 3

Уравнение Пуассона в трапеции. Последовательная программа, OpenMP, MPI и MPI+OpenMP.

[Отчёт по первой серии расчётов](report/report.pdf) · [Исходник LaTeX](report/report.tex)

## Запуск на Полюсе

Из каталога `task1`:

```bash
module load SpectrumMPI
make all
export RESULTS_DIR="$PWD/results/review-$(date +%Y%m%d-%H%M%S)"
bash submit.sh seq
bash submit.sh omp
bash submit.sh mpi
bash submit.sh hybrid
bjobs -a
```

Каждая команда `submit.sh` отправляет одно задание. Каждая конфигурация запускается пять раз. В задании OpenMP обе большие сетки дополнительно считаются последовательной программой и OpenMP с одним потоком на одном ядре.

В каталогах `run1`–`run5` лежат результаты отдельных запусков. В `timings.txt` собраны минимум, медиана и максимум времени в секундах. В `openmp_speedup.txt` приведены ускорения относительно последовательной программы и OpenMP с одним потоком на том же узле. Привязка потоков перед замером записана в `*_placement.txt`.

Когда все четыре задания получат статус `DONE`, архив для отчёта можно собрать в той же сессии:

```bash
tar -czf task1-review.tar.gz \
    --exclude='400x600*_solution.dat' \
    --exclude='800x1200*_solution.dat' \
    -C "$RESULTS_DIR" .
```
