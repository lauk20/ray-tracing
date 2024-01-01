#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

/*
    convert degrees to radians

    @param degrees the angle

    @return the angle in radians
*/
inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

/*
    get random double in [0, 1)

    @return double in [0, 1)
*/
__device__ inline float random_double(curandState * state) {
    return curand_uniform(state);
}

/*
    get random double in [min, max)

    @return double in [min, max)
*/
__device__ inline float random_double(float min, float max, curandState * state) {
    return min + (max-min) * curand_uniform(state);
}

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif