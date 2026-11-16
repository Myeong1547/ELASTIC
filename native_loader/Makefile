CC       ?= gcc
CFLAGS   ?= -O2 -g -Wall -Wextra -Wno-unused-parameter -fPIC -pthread
LIBURING ?= -luring

SRC := src
BIN := build

OBJS    := $(BIN)/loader.o
SHARED  := $(BIN)/libnvme_loader.so

.PHONY: all clean

all: $(SHARED)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/%.o: $(SRC)/%.c $(SRC)/loader.h | $(BIN)
	$(CC) $(CFLAGS) -I$(SRC) -c $< -o $@

$(SHARED): $(BIN)/loader.o
	$(CC) -shared -o $@ $^ $(LIBURING) -lpthread -ldl

clean:
	rm -rf $(BIN)
