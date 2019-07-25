//
//  buffer.h
//  BinaryClassification
//
//  Created by Naga V Gudapati on 6/13/19.
//  Copyright © 2019 Naga V Gudapati. All rights reserved.
//

#ifndef buffer_h
#define buffer_h

double k1 = 3.0;
double k2 = 2.0;

//Using a lambda to get the
std::transform(grad.begin(), grad.end(), x.begin(), grad.begin(),[&k1, &k2](auto gn, auto xn){return k1*gn + k2*xn;});

//    std::vector<double> x(M.n, 0);
//    x[0] = 0;
//    x[1] = 0;
//    x[2] = 0;
//    x[3] = 0;
//    x[4] = 0;
//
//    std::vector<double> ATx(M.m, 0);
//
////    print_vector(x);
//
//    double lambda = 2.0;
//
//    LogLoss loss_func;
//
//    loss_func.compute_data_times_vector(M, x, ATx);
//
//
//    print_vector(ATx);
//
//    std::cout << loss_func.compute_obj_val(ATx, M, x, lambda) << "\n";
//
//    std::vector<double> grad(M.n, 0);
//
//    loss_func.compute_grad_at_x(ATx, M, x, lambda, grad);
//
//    std::cout << "printting gradient : "<< "\n";
//    print_vector(grad);
//
//    return 0;


#endif /* buffer_h */
