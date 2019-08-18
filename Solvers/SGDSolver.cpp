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
    lambda = 1.0/A.n;
    eta = 1.0;
    epochs = max_iter;
    
    auto seed = rd();
    rand_gen = std::mt19937(seed);
    distribution = std::uniform_int_distribution<int>(0, A.m-1);
    
}

void SGDSolver::run_solver(const Classification_Data_CRS &A){
    
    // At the begenning, we just have the x varibale which will be updated in each epoch and within each epoch,
    // the x variable will be updated at each sample that has been analysed.
    
    //Let us first set up the variables
    log_loss.compute_data_times_vector(A, x, ATx);
    double obj_val =  log_loss.compute_obj_val(ATx, A, x, lambda);
    
    //    log_loss.compute_grad_at_x(ATx, A, x, lambda, grad);  //This computes the complete gradient (i.e for all obsvs)
    
    double train_error = log_loss.compute_training_error(A, ATx);
    log_loss.compute_grad_at_x(ATx, A,x, lambda, grad);
    
    //Setting up the output that would be visible on screen"
    std::cout << "   Iter  " <<  "    obj. val  " << "     training error "  << "        Gradient Norm  " << "\n";
    std::cout << std::setw(10) << std::left << 0 << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error << std::setw(20) << std::left << get_vector_norm(grad)<< "\n";
    
    // Now we have to update x.                           L
    //    std::transform(x.begin(), x.end(), grad.begin(), x.begin(), [=](double x_i, double grad_i){return x_i - this->eta*grad_i;});
    
    //Now let us run the solver for all the epochs
    for(int t = 1; t <= this->epochs; t++){
        this->run_one_stochastic_epoch(A, x, t);
        log_loss.compute_data_times_vector(A, x, ATx);
        obj_val =  log_loss.compute_obj_val(ATx, A, x, lambda);
        train_error = log_loss.compute_training_error(A, ATx);
        log_loss.compute_grad_at_x(ATx, A,x, lambda, grad);
        std::cout << std::setw(10) << std::left << 0 << std::setw(20) << std::left << std::setprecision(10) << obj_val << std::setw(20) << std::left << train_error << std::setw(20) << std::left << get_vector_norm(grad)<< "\n";

    }
    
}


void SGDSolver::run_one_stochastic_epoch(const Classification_Data_CRS &A, std::vector<double> &x, int epoch_counter){
    
    //    The pseudocode for running one epoch is:
    //    input x_t  (at t epoch)
    //    set x <- x_t
    //    for it = 1, ... , m do:  //There are m samples
    //        choose i in (1,..,m) randomly
    //        compute the gradient at the single ( y_i, a_i ) observation               //  Step 0
    //        set η = 1/(1 + t) // t^th epoch and m samples and i^th sample             //  Step 1
    //        update x = x - η(g + λx)                                                  //  Step 2
    //    x_t+1 <- x_t
    
    
    //    It is important to realzie that step 2 changes all the coordinates of x and for very sparse x,
    //    the gradient is gonna be sparse and we have to do a sparse update.
    
    auto scale = 1.0;
    
    for ( auto it = 0; it < A.m; it++){  //iterating through the m samples
        //Let us get a random integer between 0 and m
        auto random_obs_index = distribution(rand_gen) ; //Store the observation number which is going to be used.
        
        //Step 0. The stochastic gradient is going to be be computed here itself.
        //The  gradient  L'(x; a_i, y_i) at (a_i, y_i) looks like this
        //
        
        
        //                                                 [  ai_1*y   ]
        //                                                 [  ai_2*y   ]
        //                                 (-y*aTx)        [  ai_3*y   ]
        //                                e                [    .      ]
        //         L'(x; a_i,y_i) = ------------------     [    .      ]
        //                                    (-y*aTx)     [    .      ]
        //                               1 + e             [    .      ]
        //                                                 [    .      ]
        //                                                 [  ai_n*y   ]
        //
        //                          :__________________:  :______________:
        //                              factor_1             sparse a_i
        //
        
        //        To compute this gradient, the quickest way I can think of now is to loop though sparse a_i twice
        
        
        auto y = A.y_label[random_obs_index];  //Stores the y value of that random_observation chosen
        // To calculate the  factor_1
        auto rand_aTx = 0.0;
        auto factor_1 = 0.0;
        for(auto i = A.row_ptr[random_obs_index]; i < A.row_ptr[random_obs_index+1]; i++){
            rand_aTx += A.values[i]*x[A.col_index[i]];
        }
        factor_1 = (-1.0*y)/(1.0+exp(y*rand_aTx));
        grad.resize(A.n, 0);  //Most likely not needed
        for (auto i = A.row_ptr[random_obs_index]; i < A.row_ptr[random_obs_index+1]; i++){
            grad[A.col_index[i]] = factor_1*A.values[i];
        }
        // We have the gradient values in the respective coordinates. now let's calculate the eta.
        eta = 1.0/(1.0+epoch_counter);
        
        // The update x = x - η(g + λx)   is slightly tricky. We can write it as follows:
        // x = x -ηλx -ηg => x = (1 -ηλ)x -ηg. On face of it, it is changing every single coordinate of x and can not do a sparse update. But we can use
        // a scaling trick to make a sparse update.  The following matlab code gives an idea how to do it. We choose a scale 's' and rescale 'g' using 's' and update the
        // scale at every iteration. Then we will multiply the final w with the updated scale.
        
        //                    Changing all coordinates
        //                    w = [0 0 0 0 0]';
        //                    g1 = [0 1 0 1 0]';
        //                    g2 = [0 0 1 0 1]';
        //                    g3 = [1 0 1 0 0]';
        //                    lam = 2;
        //                    n1 = 0.2;  n1 is the eta in first iteration
        //                    n2 = 0.3;  n2 is the eta in second iteration etc.
        //                    n3 = 0.4;
        //
        //                    w = w-n1*(g1+lam*w);  // updating all the coordinates 1st iter
        //                    w = w-n2*(g2+lam*w);  // updating all the coordinates 2nd iter
        //                    w = w-n3*(g3+lam*w);  // updating all the coordinares 3rd iter
        //                    w_all = w;
        //
        //                    sparse update with scaling
        //                    w = [0 0 0 0 0]';
        //                    s = 1;   %scale
        //
        //                    s = s*(1-n1*lam);
        //                    w(2) = w(2) - (n1/s)*g1(2);    //We know that only 2 and 4 cooridnates are non-zero in gradient/
        //                    w(4) = w(4) - (n1/s)*g1(4);
        //
        //                    s = s*(1-n2*lam);
        //                    w(3) = w(3) - (n2/s)*g2(3);
        //                    w(5) = w(5) - (n2/s)*g2(5);
        //
        //                    s = s*(1-n3*lam);
        //                    w(1) = w(1) - (n3/s)*g3(1);
        //                    w(3) = w(3) - (n3/s)*g3(3);
        //
        //                    s*w
        //                    w_cor
        
        scale = scale*(1-lambda*eta); //This will get uodated with every sample as eta chnages with every random sample read
        //        std::cout << "scale: " << scale << " " << lambda << " " <<eta << "\n";
        
        //We use the good ol' sparse updating to do the job of updating the x value at only those indices where the gradient has been changed.
        
        for( auto j = A.row_ptr[random_obs_index]; j < A.row_ptr[random_obs_index+1]; j++){
            x[A.col_index[j]] = x[A.col_index[j]] - (eta/scale)*grad[A.col_index[j]];
        }
    }
    
    
    //The scale gets updated m times s = (1-n1*lam)*(1-n2*lam)*...(1-nm*lam). andnow we have to multiple x with this number.
    std::transform(x.begin(), x.end(), x.begin(),
                   std::bind(std::multiplies<double>(), std::placeholders::_1, scale));
    
    
}


void SGDSolver::run_one_iter(const Classification_Data_CRS &A, std::vector<double>& x, std::vector<double>& ATx, std::vector<double>& grad, int iter_counter){
    (void) A;
    (void) x;
    (void) ATx;
    (void) grad;
    (void) iter_counter;
}
