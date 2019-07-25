//
//  main.cpp
//  ParseSVM
//
//  Created by Naga V Gudapati on 1/10/19.
//  A Simple library to read the libsvm files and return a sparse matrix
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#include <iostream>
#include <string>
#include "ParseSVM.hpp"
#include "Matrix_Operations.hpp"

int main(int argc, const char * argv[]) {
    
    const std::string file_path = argv[1];
    
    CCS_Matrix M;
    
    get_CCSM_from_svm(M, file_path);
    
//    print_vector(M.y_label);
//    count_labels(M.y_label);
//    std::cout << "Num observations: " << M.m << std::endl;
//    std::cout << "Num features: " << M.n << std::endl;
    return 0;
}
