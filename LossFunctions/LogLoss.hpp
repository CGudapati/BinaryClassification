//
//  LogLoss.hpp
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/11/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef LogLoss_hpp
#define LogLoss_hpp

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <functional>
#include <algorithm>

#include "../ParseSVM/Matrix.h"
#include "../helper.h"

class LogLoss {
public:
    LogLoss() {}
    
    void compute_data_times_vector(const Classification_Data_CRS&  A, const std::vector<double>& x, std::vector<double> & ATx);
    
    double compute_obj_val(const std::vector<double>& ATx, const Classification_Data_CRS& A, const std::vector<double>& x,  double lambda);
    
    void compute_grad_at_x(const std::vector<double>& ATx, const Classification_Data_CRS& A, const std::vector<double>& x,  double lambda, std::vector<double>& grad);
  
    double compute_training_error(const Classification_Data_CRS& A, const std::vector<double>& ATx);
};


#endif /* LogLoss_hpp */
