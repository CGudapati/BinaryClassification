//
//  LogisticLoss.h
//  BinaryClassification
//
//  Created by Naga V Gudapati on 8/20/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef LogLoss_h
#define LogLoss_h

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <functional>
#include <algorithm>

#include "../ParseSVM/Matrix.h"
#include "../helper.h"

namespace LogLoss {
    
    
    inline void compute_data_times_vector( const Classification_Data_CRS&  A, const std::vector<double>& x, std::vector<double> & ATx){
        
        std::fill(ATx.begin(), ATx.end(), 0.0);  //Setting the ATx vector to 0;
        
        //The number of elements in ATX is the same as number of observations in the original data.
        
        for(auto i = 0; i < A.m; ++i){
            for( auto j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j){
                ATx[i] += A.values[j]*x[A.col_index[j]];
            }
        }
        
    }
    
    
    inline double compute_obj_val(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double>& x,  double lambda){
        
        //The objective function is
        
        //      1    m             -yi*ai*x      λ       2
        //     --- * Σ  log(1 + e ^       )   + -—-*||x||
        //      m   i=1                          2
        //  where ai is the i_th row of the A matrix
        
        //Set everything to 0
        double obj_val = 0.0;
        
        //calculating the term 1
        for(auto i = 0; i < A.m; i++){  //This is the Σ_i=1..m
            obj_val += log(1+exp(-1*A.y_label[i]*ATx[i]));
        }
        
        obj_val /= A.m;  //dividing by 1/m
        
        obj_val += 0.5*lambda*l2_norm_sq(x);
        
        return obj_val;
    }
    
    
    inline void compute_grad_at_x(const std::vector<double> &ATx, const Classification_Data_CRS &A, const std::vector<double> &x, double lambda, std::vector<double> &grad){
        
        // Set the gradient to zero as you have to caclulate it from scratch. The gradient is a vector of n (number of features) elements.
        // We loop through each row in the Data matrix and update the correspondong coordinates of the gradient.
        // i.e say the first row of the data matrix has non zeros in a11, a13 and a1n
        // positions: we then update the graident in only those three corordinates.
        // After we are though all the rows, then we will scale it by m and then add the "x" vector.
        
        std::fill(grad.begin(), grad.end(), 0.0);  //Setting thegradient vector to 0;
        
        //We go through each row in the A matrix and then update the corresponding coordinates in the gradient. We are not buildning corordinate by corrdinate gradient.
        for (auto i = 0; i < A.m; i++) {
            auto accum = exp(-1*A.y_label[i]*ATx[i]);
            accum = accum/(1.0 + accum);
            for(auto j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j){
                auto temp = -1.0*accum*A.values[j]*A.y_label[i];
                grad[A.col_index[j]] += temp;
            }
        }
        
        //                                                    1                                               1
        //Using c++ lambdas to scale the current gradient by ---, scale x by λ and add them together: grad = --- * gradient +  λ*x
        //                                                    m                                               m
        // This is a classic daxpy operation
        
        //This finishes the gradient
        
        std::transform(grad.begin(), grad.end(), x.begin(), grad.begin(),[&A, &lambda](auto grad_i, auto x_i){return (1.0/A.m)*grad_i + lambda*x_i;});
        
    }
    
    inline double compute_training_error(const Classification_Data_CRS& A, const std::vector<double>& ATx){
        
        // We calculate the training errors as follows
        // If a new observation is provided, we apply the logistic function
        //
        //                1
        //     f(z) = ------------
        //                   -z
        //            (1 + e^  )
        // where z = a_i*x and a_i is the feature vector of the new observation
        
        // If f(z) > 0.5, then its predicted label is +1 and f(z) < 0.5, the predicetd label is -1;
        // More information about logistic regression can be found on wikipedia
        
        //As the algorithm progresses, we hope that the training errors reduces.
        
        
        double train_error = 0.0;
        
        std::vector<int> z(ATx.size(), 0);  //This will hold the predictions
        
        double prediction = 0.0;
        int corrent_predictions = 0;
        
        for (std::size_t i =0; i < ATx.size(); ++i) {
            prediction = 1.0/(1.0 + exp(-1.0*ATx[i]));
            if (prediction >= 0.5){
                z[i] = 1;
            }
            else{
                z[i] = -1;
            }
            
            if (A.y_label[i]== z[i]) {
                corrent_predictions++;
            }
        }
        
        return 1.0-(corrent_predictions*1.0/(A.m)); // total number of correct predictions/ total number of observations.
        
        
        
        
        return train_error;
    }
    
}


#endif /* LogLoss_h */
