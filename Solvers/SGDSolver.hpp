//
//  SGDSolver.hpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 7/24/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef SGDSolver_hpp
#define SGDSolver_hpp

#include <stdio.h>
#include "CoreSolver.hpp"

#include <random>
#include <chrono>

class SGDSolver : public CoreSolver{
    
public:
    std::vector<double> x;
    std::vector<double> grad;
    std::vector<double> ATx;
    double lambda;
    double eta;
    int epochs;
    
    //Just some variable for random integer generation.
private:
     std::random_device rd;
     std::mt19937 rand_gen;
     std::uniform_int_distribution<int> distribution;
public:
    virtual void init(const Classification_Data_CRS &A, double lam, double alfa, int max_iter);
    virtual void run_solver(const Classification_Data_CRS& A);
    virtual void run_one_stochastic_epoch(const Classification_Data_CRS &A, std::vector<double>& x, int iter_counter);
    virtual void run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad);    
};


#endif /* SGDSolver_hpp */
