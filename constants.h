#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
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

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif