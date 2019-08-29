//
//  GradientDescent.cpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/15/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

    // let f(x) be a convex, smooth and differentiable function. The following general steps will get you the optimal solution.
    // Given f(x) and x_0;
    // Step 1:  find a descent direction (search direction) "s"
    //          If no such direction exixts, STOP!!!
    // Step 2: Line Search: find an appropriate step size, α, such that arg_min f(x + α*s);
    // Step 3: Update x := x + α*s
    // Step 4: check for stopping conditions; if they are met, then you can stop. if not continue steps 1-4;



#include "GradientDescent.hpp"

void GradientDescent::init(const Classification_Data_CRS &A, double lam, double alfa, int max_iter){
    x.resize(A.n, 0); //We are creating a vector of all 0s of size n (number of features)
    grad.resize(A.n, 0); //The gradient also is of size n
    ATx.resize(A.m, 0); //This vector holds the value of A*x. (T is times and not transpose)
    lambda = lam;
    alpha = alfa;
    iters = max_iter;
}

void GradientDescent::run_solver(const Classification_Data_CRS &A){
    //Let us first set up the variables
    LogLoss::compute_data_times_vector(A, x, ATx);
    double obj_val =  LogLoss::compute_obj_val(ATx, A, x, lambda);
    double train_error = LogLoss::compute_training_error(A, ATx);
    
    //Setting up the output that would be visible on screen"
    std::cout << "   Iter  " <<  "    obj. val  " << "     training error "  <<  "\n";
    std::cout << std::setw(10) << std::left << 0 << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error  << "\n";
    
    // Now we have to update x. From the general algorithm, we have to find a descent direction,  perform a line search to get
    // the appropriate value of α and the update the value of x. In gradient descent, the negative gradient vector will
    // always give you the descent direction.  We will use a constant step size of 1/L (L is Lipschiz constant)
    //                                                      1
    // We will use std::transform to do this job: x := x - ---*g; (simple daxpy operation)
    //                                                      L
    
    //Now let us run the solver for all the iterations
    for(int k = 1; k <= this->iters; k++){
        
        this->run_one_iter(A, x, ATx, grad);
        
        LogLoss::compute_data_times_vector(A, x, ATx);
        double obj_val =  LogLoss::compute_obj_val(ATx, A, x, lambda);
        double train_error = LogLoss::compute_training_error(A, ATx);
        LogLoss::compute_grad_at_x(ATx, A,x, lambda, grad);
        
        std::cout << std::setw(10) << std::left << k << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error << "\n" ;
        
    }
}

void GradientDescent::run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad){
    
    LogLoss::compute_grad_at_x(ATx, A, x, lambda, grad);
    std::transform(x.begin(), x.end(), grad.begin(), x.begin(), [=](double x_i, double grad_i){return x_i - this->alpha*grad_i;});
    
    
}
