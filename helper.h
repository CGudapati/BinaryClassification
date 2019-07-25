//
//  helper.h
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/10/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef helper_h
#define helper_h

//simple function to print the vector
template <typename T>
void print_vector(const std::vector<T> & vec){
    for(const auto & elem : vec){
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

template <typename T>
double l2_norm_sq(std::vector<T> const& u) {
    T accum = 0.0;
    
    for(auto const& elem: u) {
        accum += elem*elem;
    }
    return accum;
}


#endif /* helper_h */
