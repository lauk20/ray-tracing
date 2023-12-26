#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

/*
    convert degrees to radians

    @param degrees the angle

    @return the angle in radians
*/
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

/*
    get random double in [0, 1)

    @return double in [0, 1)
*/
inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

/*
    get random double in [min, max)

    @return double in [min, max)
*/
inline double random_double(double min, double max) {
    return min + (max-min) * random_double();
}

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif