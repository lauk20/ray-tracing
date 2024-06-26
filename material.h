#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "constants.h"
#include "hittable.h"
#include "vec3.h"

/*
    class representing a material
*/
class material {
    public:
        virtual ~material() = default;

        /*
            ray scattering/bouncing for the material
            modifies rec, attenuation, and scattered parameters.

            @param r_in the ray
            @param rec the hit record to modify
            @param attenuation how much color each bounce should retain
            @param scattered the scattered ray

            @return true on success
        */
        virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const = 0;
};

/*
    class representing a diffuse material
*/
class lambertian : public material {
    public:
        // constructor
        lambertian(const color &a) : albedo(a) {}

        bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override {
            vec3 scatter_direction = rec.normal + random_unit_vector();
            
            // scatter direction may happen to be zero if the random
            // unit vector ends up being opposite of the normal
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            
            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        }
    
    private:
        // how much color each ray bounce should retain
        color albedo;
};

/*
    class representing metal material
*/
class metal : public material {
    public:
        // constructor
        metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz * random_unit_vector());
            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);
        }
    
    private:
        // how much color each ray bounce should retain
        color albedo;
        // fuzzy reflections
        double fuzz;
};

class dielectric : public material {
    public:
        dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override {
            attenuation = color(1.0, 1.0, 1.0);
            double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = ray(rec.p, direction);

            return true;
        }

    private:
        // index of refraction
        double ir;

        // schlick's approximation for reflectance
        static double reflectance(double cosine, double ref_idx) {
            auto r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }
};

#endif