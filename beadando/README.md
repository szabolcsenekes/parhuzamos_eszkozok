# 2D hőterjedés szimuláció OpenCL és SDL2 felhasználásával

## Projekt áttekintés

A projekt egy 2 dimenziós hőterjedési szimulációt valósít meg C
nyelven.\
A számítás kétféleképpen történik: - Szekvenciális CPU megoldással -
Párhuzamos OpenCL (GPU) megoldással

A cél a két megközelítés teljesítményének összehasonlítása.\
A szimuláció valós időben jelenik meg SDL2 segítségével.

------------------------------------------------------------------------

## Célkitűzések

-   Hőterjedési modell implementálása 2D rácson\
-   Szekvenciális CPU verzió készítése\
-   Párhuzamos OpenCL verzió megvalósítása\
-   Teljesítmény összehasonlítás\
-   Interaktív megjelenítés

------------------------------------------------------------------------

## Szimuláció működése

A modell egy 2D rácson alapul, ahol minden cella egy hőmérséklet értéket
tárol.

Minden iterációban: - A cella új értéke a négy szomszéd (fel, le, bal,
jobb) átlaga\
- A szélső cellák hőmérséklete 0.0 (hideg peremfeltétel)\
- A hőforrás cellák értéke állandóan 1.0

Ez egy egyszerű hődiffúziós folyamatot modellez.

------------------------------------------------------------------------

## Párhuzamosítás

A párhuzamosítás OpenCL segítségével valósul meg.

-   Minden rácspontot egy OpenCL work-item számol\
-   A teljes rács egy 2D NDRange-ben fut\
-   A GPU egyszerre sok cellát dolgoz fel

A CPU verzió szekvenciális marad, és referenciaként szolgál.

------------------------------------------------------------------------

## Felhasznált technológiák

-   C programozási nyelv\
-   OpenCL\
-   SDL2\
-   GCC / MinGW\
-   Makefile

------------------------------------------------------------------------

## Projekt felépítése

    src/
        main.c
        grid.c
        renderer.c
        opencl_heat.c
        benchmark.c
        util.c

    include/
        grid.h
        renderer.h
        opencl_heat.h
        benchmark.h
        util.h

    kernels/
        heat_kernel.cl

    data/
        outputs.csv

------------------------------------------------------------------------

## Kezelés

-   `ESC` → kilépés\
-   `SPACE` → szünet / folytatás\
-   `R` → reset\
-   bal egérgomb → új hőforrás

------------------------------------------------------------------------

## Mérések (Benchmark)

A program automatikus méréseket végez:

-   CPU futási idő\
-   OpenCL futási idő\
-   Gyorsulás (speedup)

Az eredmények: `data/outputs.csv`

------------------------------------------------------------------------

## Példa eredmény

    CPU time: 207.765 ms
    OpenCL time: 78.333 ms
    Speedup: 2.65x

------------------------------------------------------------------------

## Fordítás és futtatás

    make
    make run

------------------------------------------------------------------------

## Átlagos teljesítmény (500 iteráció)

  Méret       CPU (ms)   OpenCL (ms)   Gyorsulás
  ----------- ---------- ------------- -----------
  128×128     \~13.4     \~179.1       0.07x
  256×256     \~51.3     \~37.6        1.36x
  512×512     \~230.0    \~110.0       \~2.1x
  1024×1024   \~834.7    \~167.2       \~4.9x

### Megfigyelések

-   Kis méret esetén a GPU lassabb a magas inicializációs költségek
    miatt\
-   Közepes mérettől kezdve a GPU előnybe kerül\
-   Nagy méretnél jelentős gyorsulás figyelhető meg

------------------------------------------------------------------------

## Iterációszám hatása (256×256)

  Iteráció   CPU (ms)   OpenCL (ms)   Gyorsulás
  ---------- ---------- ------------- -----------
  100        9.82       11.48         0.85x
  500        \~51.3     \~37.6        1.36x
  1000       99.72      66.48         1.50x
  2000       188.85     233.53        0.81x

### Megfigyelések

-   Kevés iterációnál a GPU nem hatékony\
-   Közepes iterációszámnál javul a teljesítmény\
-   Nagyon sok iterációnál a memória-másolás költsége dominálhat

------------------------------------------------------------------------

## Skálázódás (1000 iteráció)

  Méret       CPU (ms)   OpenCL (ms)   Gyorsulás
  ----------- ---------- ------------- -----------
  64×64       6.22       49.54         0.13x
  128×128     24.96      53.54         0.47x
  256×256     99.72      66.48         1.50x
  512×512     419.86     133.44        3.15x
  1024×1024   1785.21    350.62        5.09x

### Megfigyelések

-   A GPU overhead dominál kis problémaméreteknél\
-   A gyorsulás közel lineárisan nő a mérettel\
-   Nagy adathalmazok esetén a GPU jelentős előnyt biztosít

------------------------------------------------------------------------

## Legjobb mért eredmények

  Méret       CPU (ms)   OpenCL (ms)   Gyorsulás
  ----------- ---------- ------------- -----------
  1024×1024   1977.19    317.65        6.22x
  1024×1024   1963.91    321.65        6.10x
  1024×1024   1785.21    350.62        5.09x
  1024×1024   834.59     167.38        4.99x

------------------------------------------------------------------------

## Összegzés

-   A GPU kis méretű problémáknál nem hatékony\
-   A teljesítményelőny a rácsméret növekedésével nő\
-   A legnagyobb mért gyorsulás \~6x volt\
-   A memória-kezelés és adatmásolás jelentős hatással van a
    teljesítményre

------------------------------------------------------------------------

## Következtetés

Az OpenCL alapú párhuzamosítás különösen nagy méretű rácsok esetén
hatékony.\
A CPU implementáció kisebb problémák esetén versenyképesebb lehet,
azonban a GPU jelentős gyorsulást biztosít nagy számítási igény mellett.
