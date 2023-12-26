#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

/*
    vec3 class represents a vector in R3
*/
class vec3 {
    public:
        // the vector
        double m[3];

        // constructors
        vec3(): m{0, 0, 0} {}
        vec3(double m0, double m1, double m2) : m{m0, m1, m2} {}

        // accessers
        double x() const {
            return m[0];
        }
        double y() const {
            return m[1];
        }
        double z() const {
            return m[2];
        }

        // operator overloads

        // negate the vector
        vec3 operator-() const {
            return vec3(-m[0], -m[1], -m[2]);
        }
        // access vector elements
        double operator[](int i) const {
            return m[i];
        }
        // access vector elements
        double& operator[](int i) {
            return m[i];
        }

        // vector addition to existing vector
        vec3& operator+=(const vec3 &v) {
            m[0] += v.m[0];
            m[1] += v.m[1];
            m[2] += v.m[2];

            return *this;
        }

        // scalar multiply existing vector
        vec3& operator*=(double t) {
            m[0] *= t;
            m[1] *= t;
            m[2] *= t;

            return *this;
        }

        // scalar divide existing vector
        vec3& operator/=(double t) {
            return *this *= 1/t;
        }

        // size of vector
        double length() const {
            return std::sqrt(length_squared());
        }

        // size of vector squared
        double length_squared() const {
            return m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        }

        // generate random vector3
        static vec3 random() {
            return vec3(random_double(), random_double(), random_double());
        }

        /*
            generate random vec3 with min, max

            @param min the minimum
            @param max the maximum

            @return random vec3
        */
        static vec3 random(double min, double max) {
            return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
        }
};

// point3 will be alias for vec3
// the point will be in the R3 space
using point3 = vec3;

// vector utility functions
// mainly vector operations

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.m[0] << " " << v.m[1] << " " << v.m[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.m[0] + v.m[0], u.m[1] + v.m[1], u.m[2] + v.m[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.m[0] - v.m[0], u.m[1] - v.m[1], u.m[2] - v.m[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.m[0] * v.m[0], u.m[1] * v.m[1], u.m[2] * v.m[2]);
}

inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t * v.m[0], t * v.m[1], t * v.m[2]);
}

inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

inline vec3 operator/(const vec3 &v, double t) {
    return (1/t) * v;
}

// dot product
inline double dot(const vec3 &u, const vec3 &v) {
    return u.m[0] * v.m[0] + u.m[1] * v.m[1] + u.m[2] * v.m[2];
}

// cross product
inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.m[1] * v.m[2] - u.m[2] * v.m[1],
                u.m[2] * v.m[0] - u.m[0] * v.m[2],
                u.m[0] * v.m[1] - u.m[1] * v.m[0]);
}

// get unit vector
inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

// get random vector within unit sphere
inline vec3 random_in_unit_sphere() {
    while (true) {
        vec3 p = vec3::random(-1, 1);
        if (p.length_squared() < 1)
            return p;
    }
}

// get random unit vector within unit sphere
inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

/*  determine whether a vector is in the same hemisphere of the normal
    (whether a vector points outward of the sphere or not)
    if it is not, return the inverted vector

    @param normal surface normal of sphere, pointing outward
    
    @return random unit vector that points outward
*/
inline vec3 random_on_hemisphere(const vec3 &normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

#endif