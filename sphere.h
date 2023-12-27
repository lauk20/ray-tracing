#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
    public:
        // constructor
        sphere(point3 _center, double _radius, shared_ptr<material> _material) : center(_center), radius(_radius), mat(_material) {}
        
        /*
            determine whether the ray hits the sphere within ray_tmin and ray_tmax

            Derived from the equation (x-C_x)^2 + (y-C_y)^2 + (z-C_z)^2 = r^2
            and then representing the equation using vectors where the center is
            C = (C_x, C_y, C_z) and the point on the sphere is P = (x, y, z).
            The dot prouct of (P-C) and (P-C) = the initial equation.
            So, (P-C) DOT (P-C) = r^2.
            Then using the fact that our ray can be represented using the equation
            P(t) = A + tb, we plug in and solve for t to determine the number of
            solutions (intersections of the ray and sphere).

            @param r the ray
            @param ray_tmin the minimum parameter t for a valid hit
            @param ray_tmax the maximum parameter t for a valid hit
            @param rec the hit_record to modify upon valid hit

            @return true if valid hit, false otherwise
        */
        bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
            // calculate ray-sphere hit points for parameter t
            vec3 center_to_origin = r.origin() - center;
            double a = r.direction().length_squared();
            double half_b = dot(r.direction(), center_to_origin);
            double c = center_to_origin.length_squared() - radius * radius;
            double discriminant = half_b * half_b - a * c;

            if (discriminant < 0) return false;
            double sqrt_discriminant = std::sqrt(discriminant);

            double root = (-half_b - sqrt_discriminant) / a;
            if (!ray_t.surrounds(root)) {
                root = (-half_b + sqrt_discriminant) / a;
                if (!ray_t.surrounds(root)) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat = mat;

            return true;
        }
    
    private:
        // center of sphere
        point3 center;
        // radius of sphere
        double radius;
        // material of the sphere
        shared_ptr<material> mat;
};

#endif