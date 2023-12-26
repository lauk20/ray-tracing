#ifndef INTERVAL_H
#define INTERVAL_H

extern const double infinity;

/*
    class representing a real-valued interval
*/
class interval {
    public:
        // min, max of the interval
        double min, max;

        // constructors
        interval() : min(infinity), max(-infinity) {}
        interval(double _min, double _max) : min(_min), max(_max) {}

        /*
            determine whether x is contained in the interval (inclusive)

            @param x the value

            @return true if min <= x <= max; false otherwise
        */
        bool contains(double x) const {
            return min <= x && x <= max;
        }

        /*
            determine whether x is contained in the interval (exclusive)

            @param x the value

            @return true if min < x < max; false otherwise
        */
        bool surrounds(double x) const {
            return min < x && x < max;
        }

        /*
            mathematical clamp function that clamps to min, max

            @param x number to clamp

            @return the clamped number
        */
        double clamp(double x) const {
            if (x < min) return min;
            if (x > max) return max;

            return x;
        }

        static const interval empty, universe;
};

const static interval empty(infinity, -infinity);
const static interval universe(-infinity, infinity);

#endif