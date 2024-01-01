#ifndef INTERVAL_H
#define INTERVAL_H

extern const float infinity;

/*
    class representing a real-valued interval
*/
class interval {
    public:
        // min, max of the interval
        float min, max;

        // constructors
        __host__ __device__ interval() : min(infinity), max(-infinity) {}
        __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

        /*
            determine whether x is contained in the interval (inclusive)

            @param x the value

            @return true if min <= x <= max; false otherwise
        */
        __host__ __device__ bool contains(float x) const {
            return min <= x && x <= max;
        }

        /*
            determine whether x is contained in the interval (exclusive)

            @param x the value

            @return true if min < x < max; false otherwise
        */
        __host__ __device__ bool surrounds(float x) const {
            return min < x && x < max;
        }

        /*
            mathematical clamp function that clamps to min, max

            @param x number to clamp

            @return the clamped number
        */
        __host__ __device__ float clamp(float x) const {
            if (x < min) return min;
            if (x > max) return max;

            return x;
        }

        static const interval empty, universe;
};

const static interval empty(infinity, -infinity);
const static interval universe(-infinity, infinity);

#endif