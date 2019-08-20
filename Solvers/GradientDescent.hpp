//
//  GradientDescent.hpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/15/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef GradientDescent_hpp
#define GradientDescent_hpp

#include <stdio.h>
#include "CoreSolver.hpp"

class GradientDescent : public CoreSolver{
public:
    
    std::vector<double> x;
    std::vector<double> grad;
    std::vector<double> ATx;
    double lambda;
    double alpha;
    int iters;
    
    virtual void init(const Classification_Data_CRS &A ,double lam, double alfa, int max_iter);
    virtual void run_solver(const Classification_Data_CRS& A);
    virtual void run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad);
    
    
};

#endif /* GradientDescent_hpp */
