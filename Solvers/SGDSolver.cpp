//
//  SGDSolver.cpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 7/24/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#include "SGDSolver.hpp"

//virtual void init(const Classification_Data_CRS &A, double lam, double α, int max_iter);

void SGDSolver::init(const Classification_Data_CRS &A, double lam, double α, int max_iter){
   
    x.resize(A.n, 0); //We are creating a vector of all 0s of size n (number of features)
    grad.resize(A.n, 0); //The gradient also is of size n
    ATx.resize(A.m, 0); //This vector holds the value of A*x. (T is times and not transpose)
    lambda = lam;
    eta = α;
    epochs = max_iter;
    
//    auto seed = rd();
//    rand_gen = std::mt19937(seed);
//    std::uniform_int_distribution<int> distribution(0, A.m-1);
    
    
    ////    this->rd;
//    rng.seed(::time(NULL));
//    std::uniform_int_distribution<int> dist (0, A.m-1);
//    this->rng(this->rd());
}

void SGDSolver::run_solver(const Classification_Data_CRS &A){
    
    // At the begenning, we just have the x varibale which will be updated in each epoch and within each epoch,
    // the x variable will be updated at each sample that has been analysed.
    
    //Let us first set up the variables
    log_loss.compute_data_times_vector(A, x, ATx);
    double obj_val =  log_loss.compute_obj_val(ATx, A, x, lambda);
    
    //    log_loss.compute_grad_at_x(ATx, A, x, lambda, grad);  //This computes the complete gradient (i.e for all obsvs)
    
    double train_error = log_loss.compute_training_error(A, ATx);
    //Setting up the output that would be visible on screen"
    std::cout << "   Iter  " <<  "    obj. val  " << "     training error " << "\n";
    std::cout << std::setw(10) << std::left << 0 << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error << "\n";
    
    // Now we have to update x.                           L
    //    std::transform(x.begin(), x.end(), grad.begin(), x.begin(), [=](double x_i, double grad_i){return x_i - this->eta*grad_i;});
    
    //Now let us run the solver for all the epochs
    for(int k = 1; k <= this->epochs; k++){
        
        this->run_one_stochastic_epoch(A, x, k);
        
        
    }
}


void SGDSolver::run_one_stochastic_epoch(const Classification_Data_CRS &A, std::vector<double> &x, int iter_counter){
    
    //    The pseudocode for running one epoch is:
    //    input x_t  (at t epoch)
    //    set x <- x_t
    //    for it = 1, ... , m do:  //There are m samples
    //        choose i in (1,..,m) randomly
    //        compute the gradient at the single ( y_i, a_i ) observation               //  Step 0
    //        set η = 1/(1 + t*m + i) // t^th epoch and m samples and i^th sample       //  Step 1
    //        update x = x - η(g + λx)                                                  //  Step 2
    //    x_t+1 <- x_t
    
    
    //    It is important to realzie that step 2 changes all the coordinates of x and for very sparse x,
    //    the gradient is gonna be sparse and we have to do a sparse update. There are some tricks available
    //    to do it but I will be doing something slightly different.
    
    for ( auto it = 0; it < A.m; it++){  //iterating through the m samples
        //Let us get a random integer between 0 and m
        
        std::cout << "yay" << "\n";
        
        
    }
    
    
}


void SGDSolver::run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad, int iter_counter){
}
