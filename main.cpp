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

int main() {
    
    const std::string file_path = "/Users/cgudapati/Research/BinaryClassification/data/enron.libsvm";
    
    Classification_Data_CRS A;
    
    
    //We will store the problem data in variable A and the data is going to be normalized
    get_CRSM_from_svm(A, file_path);
    
//    std::cout <<  "The  first index of row ptr is " << A.row_ptr[10] << std::endl;
//    std::cout <<  "The  first index  of colid is " << A.col_index[10000] << std::endl;
//    std::cout <<  "The  first index  of values is " << A.values[10000] << std::endl;

    
//    print_vector(M.col_index);
//    print_vector(M.row_ptr);
//    print_vector(A.values);
//    print_vector(M.y_label);
//    std::cout << "Num features: " << M.n << "\n";
    
//    LogLoss  log_loss;
    std::cout << "GD: " << "\n";
    GradientDescent GD;
    double lambda = 0.0001;
    GD.init(A, lambda, 10, 100);
    GD.run_solver(A);
    
    std::cout << "SGD: " << "\n";
    SGDSolver SGD;
//    lambda = 0.001;
    SGD.init(A, 0.001, 10, 100);
    SGD.run_solver(A);
    
    
    
    
}
