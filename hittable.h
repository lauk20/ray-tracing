#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

/*
    structure to store ray hits
*/
class hit_record {
    public:
        // the hit point
        point3 p;
        // surface normal of the hit point
        vec3 normal;
        // parameter t of the ray equation P = A + Bt
        double t;
        // whether the ray hit the object from the inside or outside
        // true if outside, false if inside
        bool front_face;

        /*
            Set the **SURFACE** normal of the hit point.
            The outward normal is calculated from an object's hit
            function. 
            As implied, it always points from the object's inside
            out towards the hit point.
            This function determines the surface normal, which determines
            which side the ray hit the object's surface (inside or outside).
            The dot product of the ray and the outward normal is negative if
            the ray hit the outside surface, positive if the ray
            hit the object on the inside surface.
            We use an implementation where the surface normal points
            inward if the ray hit on the inside, points outward if the
            ray hit on the outside of the object.

            @param r the ray that hit the point
            @param outward_normal the 
        */
        void set_face_normal(const ray &r, const vec3 &outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

/*
    abstract class representing a hittable object
*/
class hittable {
    public:
        virtual ~hittable() = default;

        virtual bool hit(const ray &r, double ray_tmin, double ray_tmax, hit_record &rec) const = 0;
};

#endif