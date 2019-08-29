//
//  CoreSolver.cpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/10/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#include "CoreSolver.hpp"

double CoreSolver::get_vector_norm(const std::vector<double>& v){
    double accum = 0.0;
    
    for( double x : v){
        accum += x*x;
    }
    return sqrt(accum);
}
