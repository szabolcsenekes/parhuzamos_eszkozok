CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lOpenCL -lSDL2 -lm

TARGET = heat_sim
SRC = src/main.c src/grid.c src/renderer.c src/util.c src/benchmark.c src/opencl_heat.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	.\$(TARGET).exe

clean:
	del /Q src\*.o 2>nul
	del /Q $(TARGET).exe 2>nul

rebuild: clean all