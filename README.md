# 2D hőterjedés szimuláció OpenCL és SDL2 felhasználásával

## Projekt áttekintés

A projekt egy 2 dimenziós hőterjedési szimulációt valósít meg C
nyelven.
A számítás kétféleképpen történik:

- Szekvenciális CPU megoldással
- Párhuzamos OpenCL (GPU) megoldással

A cél a két megközelítés teljesítményének összehasonlítása.
A szimuláció valós időben jelenik meg SDL2 segítségével.

------------------------------------------------------------------------

## Célkitűzések

-   Hőterjedési modell implementálása 2D rácson
-   Szekvenciális CPU verzió készítése
-   Párhuzamos OpenCL verzió megvalósítása
-   Teljesítmény összehasonlítás
-   Interaktív megjelenítés

------------------------------------------------------------------------

## Szimuláció működése

A modell egy 2D rácson alapul, ahol minden cella egy hőmérséklet értéket
tárol.

Minden iterációban:

- A cella új értéke a négy szomszéd átlaga
- A szélső cellák hőmérséklete 0.0
- A hőforrás cellák értéke állandóan 1.0

Ez egy egyszerű hődiffúziós folyamatot modellez.

------------------------------------------------------------------------

## Párhuzamosítás

A párhuzamosítás OpenCL segítségével valósul meg.

-   Minden rácspontot egy OpenCL work-item számol
-   A teljes rács egy 2D NDRange-ben fut
-   A GPU egyszerre sok cellát dolgoz fel

A CPU verzió szekvenciális marad, és referenciaként szolgál.

------------------------------------------------------------------------

## Felhasznált technológiák

-   C programozási nyelv
-   OpenCL
-   SDL2
-   GCC / MinGW
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

-   `ESC` → kilépés
-   `SPACE` → szünet / folytatás
-   `R` → reset
-   bal egérgomb → új hőforrás

------------------------------------------------------------------------

## Mérések (Benchmark)

A program automatikus méréseket végez:

-   CPU futási idő
-   OpenCL futási idő
-   Gyorsulás (speedup)

Az eredmények: `data/outputs.csv`

------------------------------------------------------------------------

## Fordítás és futtatás

    make
    make run

------------------------------------------------------------------------

## Átlagos teljesítmény (500 iteráció)

| Méret       | CPU (ms) | OpenCL total (ms) | Compute (ms) | Gyorsulás |
|------------|---------|------------------|-------------|----------|
| 64×64       | ~3.5    | ~54              | ~54         | 0.06x    |
| 128×128     | ~15     | ~59              | ~59         | 0.25x    |
| 256×256     | ~57     | ~67              | ~67         | 0.85x    |
| 512×512     | ~229    | ~96              | ~95         | 2.4x     |
| 1024×1024   | ~915    | ~272             | ~270        | 3.3–4.3x |

------------------------------------------------------------------------

## Megfigyelések

- Kis rácsméret esetén a GPU jelentősen lassabb a CPU-nál.
- Ennek oka a kernel indítási és adatmozgatási overhead.
- Közepes méretnél (256×256) a teljesítmény közel azonos.
- Nagyobb méretnél (512×512 felett) a GPU egyértelmű előnybe kerül.
- A maximális gyorsulás ~4x körül alakult.

### Memória másolás hatása

A mérések alapján:

- Az adatmásolás (upload + download) viszonylag kicsi időt vesz igénybe
  (~1–3 ms nagy méreteknél).
- A teljes futási időt főként a kernel futás (compute) dominálja.
- A compute-only gyorsulás valamivel nagyobb, mint a teljes gyorsulás.

Ez azt mutatja, hogy:
> a jelen implementációban a GPU számítás dominál, nem az adatmásolás.

### Skálázódás

- A CPU futási ideje közel lineárisan nő a rács méretével.
- A GPU futási ideje lassabban nő.
- Emiatt a gyorsulás a mérettel együtt növekszik.

Ez megfelel az elméleti várakozásoknak.

------------------------------------------------------------------------

## OpenCL futás felbontása

A GPU futási idő három részre bontható:

- Upload (CPU → GPU)
- Compute (kernel futás)
- Download (GPU → CPU)

Példa (1024×1024):

- Upload: ~2.5 ms
- Compute: ~270 ms
- Download: ~0.9 ms

Megfigyelés:

- A compute dominál (a teljes futási idő döntő részét teszi ki)
- Az adatmozgatás költsége elhanyagolható ebben a méretben

A kernel futtatása során kézi work-group méret (pl. 16×16) került
beállításra, amely stabil teljesítményt eredményezett.

------------------------------------------------------------------------

## Hardver konfiguráció

A mérések az alábbi rendszeren készültek:

- CPU: AMD Ryzen 5 5600H
- GPU: AMD Radeon(TM) Graphics (Vega, 512 cores)
- OpenCL: 1.2
- RAM: 512 MB GPU shared memory
- Driver: AMD Adrenalin 23.4.2

A GPU integrált (iGPU), ezért a memória CPU és GPU között megosztott.

------------------------------------------------------------------------

## Összegzés

- A GPU kis problémaméreteknél nem hatékony
- A teljesítményelőny a rácsméret növekedésével nő
- A maximális gyorsulás ~4x volt
- Az OpenCL implementáció jelentős gyorsulást biztosít nagy méretű rácsok esetén
- A számítási időt elsősorban a kernel futása határozza meg

------------------------------------------------------------------------

## Következtetés

Az OpenCL alapú párhuzamosítás különösen nagy méretű rácsok esetén
hatékony.
A CPU implementáció kisebb problémák esetén versenyképesebb lehet,
azonban a GPU jelentős gyorsulást biztosít nagy számítási igény mellett.
A mérési eredmények alapján megállapítható, hogy a gyorsulás mértékét
nemcsak a párhuzamos számítás, hanem a GPU kihasználtsága és a
probléma mérete is jelentősen befolyásolja.
