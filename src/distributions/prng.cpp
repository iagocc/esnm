#include "prng.h"
#include <stdint.h>

// 1. Define the internal state structure
typedef struct {
    uint64_t s[4];
} xoshiro256plusplus_state_t;

// Helper: Bitwise left rotation
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// 2. The GSL 'set' function (Seeding via SplitMix64)
static void gsl_xoshiro256plusplus_set(void *vstate, unsigned long int seed) {
    xoshiro256plusplus_state_t *state = (xoshiro256plusplus_state_t *)vstate;
    uint64_t sm_state = (uint64_t)seed;

    for (int i = 0; i < 4; i++) {
        sm_state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = sm_state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        state->s[i] = z ^ (z >> 31);
    }
}

// 3. The GSL 'get' function
static unsigned long int gsl_xoshiro256plusplus_get(void *vstate) {
    xoshiro256plusplus_state_t *state = (xoshiro256plusplus_state_t *)vstate;

    const uint64_t result = rotl(state->s[0] + state->s[3], 23) + state->s[0];
    const uint64_t t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];
    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);

    return (unsigned long int)result;
}

// 4. The GSL 'get_double' function
static double gsl_xoshiro256plusplus_get_double(void *vstate) {
    xoshiro256plusplus_state_t *state = (xoshiro256plusplus_state_t *)vstate;
    
    const uint64_t result = rotl(state->s[0] + state->s[3], 23) + state->s[0];
    const uint64_t t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];
    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);

    return (result >> 11) * 0x1.0p-53; 
}

// 5. Define the GSL RNG Type struct
static const gsl_rng_type gsl_rng_xoshiro256plusplus_type = {
    "xoshiro256++",          
    0xFFFFFFFFFFFFFFFFULL,   
    0,                       
    sizeof(xoshiro256plusplus_state_t),
    &gsl_xoshiro256plusplus_set,
    &gsl_xoshiro256plusplus_get,
    &gsl_xoshiro256plusplus_get_double
};

// 6. Assign the global pointer declared in the header
const gsl_rng_type *gsl_rng_xoshiro256plusplus = &gsl_rng_xoshiro256plusplus_type;
