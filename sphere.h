#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
    public:
        // constructor
        sphere(point3 _center, double _radius) : center(_center), radius(_radius) {}
        
        /*
            determine whether the ray hits the sphere within ray_tmin and ray_tmax
            for the roots t in the ray equation P = A + Bt.
            modified rec valid hit is found.

            @param r the ray
            @param ray_tmin the minimum parameter t for a valid hit
            @param ray_tmax the maximum parameter t for a valid hit
            @param rec the hit_record to modify upon valid hit

            @return true if valid hit, false otherwise
        */
        bool hit(const ray &r, double ray_tmin, double ray_tmax, hit_record &rec) const override {
            // calculate ray-sphere hit points for parameter t
            vec3 center_to_origin = r.origin() - center;
            double a = r.direction().length_squared();
            double half_b = dot(r.direction(), center_to_origin);
            double c = center_to_origin.length_squared() - radius * radius;
            double discriminant = half_b * half_b - a * c;

            if (discriminant < 0) return false;
            double sqrt_discriminant = std::sqrt(discriminant);

            double root = (-half_b - sqrt_discriminant) / a;
            if (root <= ray_tmin || ray_tmax <= root) {
                root = (-half_b + sqrt_discriminant) / a;
                if (root <= ray_tmin || ray_tmax <= root) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
        }
    
    private:
        // center of sphere
        point3 center;
        // radius of sphere
        double radius;
};

#endif