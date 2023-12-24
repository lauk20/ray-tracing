#ifndef RAY_H
#define RAY_H

#include "vec3.h"

/*
    class to represent a ray
*/
class ray {
    public:
        // constructors
        ray() {}
        ray(const point3 &origin, const vec3 &direction) : orig(origin), direction(direction) {}

        /*
            get origin as point3 object

            @return origin of vector
        */
        point3 origin() const {
            return orig;
        }

        /*
            get position of ray at parameter t

            @param t parameter t

            @return point of ray at t
        */
        point3 at(double t) const {
            return orig + t * direction;
        }

    private:
        // origin of the ray
        point3 orig;

        // direction of the ray
        vec3 direction;
};

#endif