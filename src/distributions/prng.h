#ifndef XOSHIRO_GSL_H
#define XOSHIRO_GSL_H

#include <gsl/gsl_rng.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const gsl_rng_type *gsl_rng_xoshiro256plusplus;

#ifdef __cplusplus
}
#endif

#endif // XOSHIRO_GSL_H
