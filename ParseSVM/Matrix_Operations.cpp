//
//  Matrix_Operations.cpp
//  ParseSVM
//
//  Created by Naga V Gudapati on 1/17/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#include "Matrix_Operations.hpp"

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


Classification_Data_CCS transpose(const Classification_Data_CRS & A )
{
    
    
    int m, n;
    long long nz;
    
    m = A.m;
    n = A.n;
    nz = A.nzmax;
    std::vector<int> w(m, 0);  //  Creating the working vector of size m. This will hold the number of elements in each
    //  column of the transposed matrix.
    
    Classification_Data_CCS C = Classification_Data_CCS(n,m,nz);
    
    
    
    for (int k = 0; k < nz; ++k)
    {
        ++w[A.col_index[k]]; //Getting the column counts i.e number of elements in each columns of the transposed matrix
    }
    
    
    cumulative_sum(C.col_ptr, w, m);
    
    
    for (int k = 0; k < n; ++k)
    {
        
        for (int p = A.row_ptr[k]; p < A.row_ptr[k+1]; ++p)
        {
            int q = w[A.col_index[p]];
            C.row_index[q] = k;
            
            C.values[q] = A.values[p];
            ++w[A.col_index[p]];
            
        }
    }
    
    C.y_label = A.y_label;
    
    return C;
    
}


