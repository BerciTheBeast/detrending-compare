# Pregled, implementacija in ovrednotenje metod za predprocesiranje ritmičnih podatkov - primerjava

Avtor: Jaka Bernard

Mentor: izr. prof. dr. Miha Moškon
Ljubljana, 2024

## How to use
Run the following command to get data used in the master's thesis locally:

```
git lfs fetch --all
git lfs pull
```

You'll need an appropriate version of CosinorPy that has all detrending methods implemented.

Then run the scripts in order:
```
process-generated.py
process-results.py
```

If you want to generate latex tables or pdf graphs you can user `generate-table.py` and `make-graphs.py`. You may need to adjust for your requirements, and possibly commend out some code that was used for generating figures for the master's thesis.