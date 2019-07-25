//
//  Matrix_Operations.hpp
//  ParseSVM
//
//  Created by Naga V Gudapati on 1/17/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef Matrix_Operations_hpp
#define Matrix_Operations_hpp

#include <stdio.h>
#include "Matrix.h"
#include <vector>
#include <iostream>

template <typename T>
void count_labels(const std::vector<T> & vec){
    int positive = 0;
    int negative = 0;
    for(auto elem : vec){
        if (elem > 0) {
            positive++;
        }
        else{
            negative++;
        }
    }
    std::cout << "pos labels: " << positive << " negative labels: " << negative << std::endl;
}



double cumulative_sum( std::vector<int> &Cp, std::vector<int> &col_count, int n);


Classification_Data_CCS transpose(const Classification_Data_CRS & A );





#endif /* Matrix_Operations_hpp */
