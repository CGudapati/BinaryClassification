//
//  ParseSVM.hpp
//  ParseSVM
//
//  Created by Naga V Gudapati on 1/11/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef ParseSVM_hpp
#define ParseSVM_hpp

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>      // std::stringstream, std::stringbuf
#include "Matrix.h"
#include "Matrix_Operations.hpp"
#include <cmath>

//   We shall try to read the libsvm file line by line and convert it into a simple compressed
//   row storage matrix. 
void get_CRSM_from_svm(Classification_Data_CRS &M, const std::string &filename);

//This funnction will used the get_CRSM_from_svm and then transpose the result
void get_CCSM_from_svm(Classification_Data_CCS &M, const std::string &filename);




#endif /* ParseSVM_hpp */
