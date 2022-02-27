CXX=clang++

HEADERS=predef.hpp loader.hpp net.hpp trainer.hpp
SRCS=main.cpp
EXE=main
INCLUDES=$(shell pkg-config eigen3 --cflags)

CXXFLAGS=-O3 -std=c++17
FLAGS=$(CXXFLAGS) $(INCLUDES)
LIBS=

OBJS=$(SRCS:%.cpp=%.o)

$(EXE): $(OBJS)
	$(CXX) $^ -o $@ $(LIBS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c $< -o $@ $(FLAGS)

clean:
	rm -f $(EXE)
	rm -f $(OBJS)

run: $(EXE)
	./$(EXE)

.PHONY: clean run
.DEFAULT_GOAL: $(EXE)
