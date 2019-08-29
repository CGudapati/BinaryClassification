//
//  main.cpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/10/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#include <iostream>
#include <string>
#include "ParseSVM/ParseSVM.hpp"
#include "ParseSVM/Matrix_Operations.hpp"
#include "helper.h"
#include "Solvers/GradientDescent.hpp"
#include "Solvers/SGDSolver.hpp"

//#include "Solvers/CoreSolver.hpp"

int main(int argc, const char * argv[]) {
    
    
    const std::string file_path = argv[1];
    
    Classification_Data_CRS A;
    
    
    //We will store the problem data in variable A and the data is going to be normalized
    get_CRSM_from_svm(A, file_path);
    
    std::cout << "GD: " << "\n";
    GradientDescent GD;
    double lambda = 0.0001;
    double Lips = 10.0;
    int iters = 100;
    GD.init(A, lambda, Lips, iters);
    GD.run_solver(A);
    
//    std::cout << "SGD: " << "\n";
//    SGDSolver SGD;
////    lambda = 0.001;
//    SGD.init(A, 0.001, 10, 100);
//    SGD.run_solver(A);
//
//
    
    
}
