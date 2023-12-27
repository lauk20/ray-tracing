#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "constants.h"
#include "hittable.h"
#include "vec3.h"

class material {
    public:
        virtual ~material() = default;

        virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const = 0;
};

class lambertian : public material {
    public:
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
        color albedo;
};

#endif