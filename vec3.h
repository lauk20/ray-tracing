#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

/*
    vec3 class represents a vector in R3
*/
class vec3 {
    public:
        // the vector
        float m[3];

        // constructors
        __host__ __device__ vec3(): m{0, 0, 0} {}
        __host__ __device__ vec3(float m0, float m1, float m2) : m{m0, m1, m2} {}

        // accessers
        __host__ __device__ float x() const {
            return m[0];
        }
        __host__ __device__ float y() const {
            return m[1];
        }
        __host__ __device__ float z() const {
            return m[2];
        }

        // operator overloads

        // negate the vector
        __host__ __device__ vec3 operator-() const {
            return vec3(-m[0], -m[1], -m[2]);
        }
        // access vector elements
        __host__ __device__ float operator[](int i) const {
            return m[i];
        }
        // access vector elements
        __host__ __device__ float& operator[](int i) {
            return m[i];
        }

        // vector addition to existing vector
        __host__ __device__ vec3& operator+=(const vec3 &v) {
            m[0] += v.m[0];
            m[1] += v.m[1];
            m[2] += v.m[2];

            return *this;
        }

        // scalar multiply existing vector
        __host__ __device__ vec3& operator*=(float t) {
            m[0] *= t;
            m[1] *= t;
            m[2] *= t;

            return *this;
        }

        // scalar divide existing vector
        __host__ __device__ vec3& operator/=(float t) {
            return *this *= 1/t;
        }

        // size of vector
        __host__ __device__ float length() const {
            return std::sqrt(length_squared());
        }

        // size of vector squared
        __host__ __device__ float length_squared() const {
            return m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        }

        // determine if the vector is close to zero in all directions
        __host__ __device__ bool near_zero() const {
            float s = 1e-8;
            return (fabs(m[0]) < s) && (fabs(m[1]) < s) && (fabs(m[2]) < s);
        }

        // generate random vector3
        __device__ static vec3 random(curandState * rand_state) {
            return vec3(random_double(rand_state), random_double(rand_state), random_double(rand_state));
        }

        /*
            generate random vec3 with min, max

            @param min the minimum
            @param max the maximum

            @return random vec3
        */
        __device__ static vec3 random(float min, float max, curandState * rand_state) {
            return vec3(random_double(min, max, rand_state), random_double(min, max, rand_state), random_double(min, max, rand_state));
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

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.m[0] + v.m[0], u.m[1] + v.m[1], u.m[2] + v.m[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.m[0] - v.m[0], u.m[1] - v.m[1], u.m[2] - v.m[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.m[0] * v.m[0], u.m[1] * v.m[1], u.m[2] * v.m[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.m[0], t * v.m[1], t * v.m[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t) {
    return (1/t) * v;
}

// dot product
__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.m[0] * v.m[0] + u.m[1] * v.m[1] + u.m[2] * v.m[2];
}

// cross product
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.m[1] * v.m[2] - u.m[2] * v.m[1],
                u.m[2] * v.m[0] - u.m[0] * v.m[2],
                u.m[0] * v.m[1] - u.m[1] * v.m[0]);
}

// get unit vector
__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

// get random vector in unit disk/circle
__device__ inline vec3 random_in_unit_disk(curandState * rand_state) {
    while (true) {
        vec3 p = vec3(random_double(-1, 1, rand_state), random_double(-1, 1, rand_state), 0);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

// get random vector within unit sphere
__device__ inline vec3 random_in_unit_sphere(curandState * rand_state) {
    while (true) {
        vec3 p = vec3::random(-1, 1, rand_state);
        if (p.length_squared() < 1)
            return p;
    }
}

// get random unit vector within unit sphere
__device__ inline vec3 random_unit_vector(curandState * rand_state) {
    return unit_vector(random_in_unit_sphere(rand_state));
}

/*  determine whether a vector is in the same hemisphere of the normal
    (whether a vector points outward of the sphere or not)
    if it is not, return the inverted vector

    @param normal surface normal of sphere, pointing outward
    
    @return random unit vector that points outward
*/
__device__ inline vec3 random_on_hemisphere(const vec3 &normal, curandState * rand_state) {
    vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

/*
    reflect a ray v, where vector n is the normal.

    @param v the ray
    @param n the normal

    @return the reflection ray
*/
__device__ inline vec3 reflect(const vec3 &v, const vec3 &n) {
    // return do * n at the end of the expression
    // because the dot product gives a scalar
    return v - 2 * dot(v, n) * n;
}

/*
    refraction using snell's law as written using vectors
    proof to be found

    @param uv the ray
    @param n the normal of the surface hit
    @param index of refraction of the outside over that of the inside object

    @return the refracted ray
*/
__device__ inline vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat) {
    float cost_theta = min(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cost_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif
