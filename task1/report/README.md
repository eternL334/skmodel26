# Отчёт к заданию 1

Отчёт: [PDF](report.pdf). Исходник: `report.tex`, рисунки: `figures/`, измерения: `measurements.json`.
Сборка через pdfLaTeX из этого каталога:

```bash
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Логотип МГУ находится в `img/msu_logo.pdf`. Дополнительные системные шрифты не нужны.

Таблица гибридной версии целиком использует повторную серию от 29.10.2026,
задание LSF 1607167. Во всех восьми расчётах число потоков подтверждено.
В measurements.json поле runs содержит использованные измерения,
superseded_runs - первоначальную гибридную серию, исключённую из таблиц.
Для повторной серии solution_compared=false: в новом архиве отсутствуют
полные поля решения, проверены журналы, невязки и история итераций.
