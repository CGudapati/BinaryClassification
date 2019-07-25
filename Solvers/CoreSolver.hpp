//
//  CoreSolver.hpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/10/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef CoreSolver_hpp
#define CoreSolver_hpp

#include <stdio.h>
#include <vector>
#include <iomanip>
#include <iomanip>

#include "../ParseSVM/Matrix.h"
#include "../LossFunctions/LogLoss.hpp"

class CoreSolver{
public:
//    CoreSolver() {}
//    virtual ~CoreSolver() {}
    virtual void init(const Classification_Data_CRS &A ,double lam, double α, int max_iter) = 0; //Implement this in the child classes
    virtual void run_solver(const Classification_Data_CRS& A) = 0; //Implement this in the child classes
    virtual void run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad, int iter_counter) = 0;
    virtual double get_vector_norm(const std::vector<double>& v);
};


#endif /* CoreSolver_hpp */
