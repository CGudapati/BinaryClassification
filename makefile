CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra 
LDFLAGS = 

all: BinaryClassification
	
run: all
	./BinaryClassification	

clean: 
	rm -f BinaryClassification *.o

BinaryClassification: main.o CoreSolver.o GradientDescent.o Matrix_Operations.o ParseSVM.o LogLoss.o SGDSolver.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

CoreSolver.o: Solvers/CoreSolver.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

GradientDescent.o: Solvers/GradientDescent.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

Matrix_Operations.o: ParseSVM/Matrix_Operations.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

ParseSVM.o: ParseSVM/ParseSVM.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

LogLoss.o: LossFunctions/LogLoss.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

SGDSolver.o: Solvers/SGDSolver.cpp
	$(CXX) $(CXXFLAGS) -c $^ $(LDFLAGS)

.PHONY: all clean
