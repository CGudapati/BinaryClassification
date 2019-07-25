//
//  Matrix_Manipulation.h
//  ParseSVM
//
//  Created by Naga V Gudapati on 1/17/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef Matrix_Manipulation_h
#define Matrix_Manipulation_h

#include "Matrix.h"

double cumulative_sum( std::vector<int> &Cp, std::vector<int> &col_count, int n  ){
    auto nz  = 0;
    
    for (auto i = 0; i < n; ++i) {
        Cp[i] = nz;
        nz += col_count[i];
        col_count[i] = Cp[i];
    }
    Cp[n] = nz;
    
    return (float)nz;
    
}


Matrix SSM_transpose(const Matrix & A )
{
    
    
    int m, n;
    long long nz;
    
    m = A.m;
    n = A.n;
    nz = A.nzmax;
    std::vector<int> w(m, 0);  //  Creating the working vector of size m. This will hold the number of elements in each
    //  column of the transposed matrix.
    
    Matrix C = Matrix(n,m,nz);
    
    
    
    for (int k = 0; k < nz; ++k)
    {
        ++w[A.col_index[k]]; //Getting the column counts i.e number of elements in each columns of the transposed matrix
    }
    
    
    cumulative_sum(C.row_ptr, w, m);
    
    
    for (int k = 0; k < n; ++k)
    {
        
        for (int p = A.row_ptr[k]; p < A.row_ptr[k+1]; ++p)
        {
            int q = w[A.col_index[p]];
            C.col_index[q] = k;
            
            if(&C.values){
                C.values[q] = A.values[p];
            }
            ++w[A.col_index[p]];
            
        }
    }
    
    return C;
    
}





#endif /* Matrix_Manipulation_h */
