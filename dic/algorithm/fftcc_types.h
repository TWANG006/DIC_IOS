//
//  types.h
//  dic
//
//  Created by Tue Le on 9/25/16.
//  Copyright © 2016 Tue Le. All rights reserved.
//

#ifndef fftcc_types_h
#define fftcc_types_h

#include "fftw3.h"

#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double intentisy_t;
typedef double real_t;
typedef fftw_plan fftw3Plan;
typedef fftw_complex fftw3Complex;
#else
typedef float intentisy_t;
typedef float real_t;
typedef fftwf_plan fftw3Plan;
typedef fftwf_complex fftw3Complex;
#endif

typedef int int_t;
typedef unsigned int uint_t;

#endif /* fftcc_types_h */
