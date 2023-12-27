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
        metal(const color &a) : albedo(a) {}

        bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected);
            attenuation = albedo;

            return true;
        }
    
    private:
        // how much color each ray bounce should retain
        color albedo;
};

#endif