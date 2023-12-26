#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>

#include "hittable.h"

using std::shared_ptr;
using std::make_shared;

/*
    class representing list of hittable objects
*/
class hittable_list : public hittable {
    public:
        // vector of hittable objects
        std::vector<shared_ptr<hittable>> objects;
        
        // constructors
        hittable_list() {}
        /*
            create a hittable list with the object in it
        */
        hittable_list(shared_ptr<hittable> object) {
            add(object);
        }

        /*
            clear the list of hittable objects
        */
        void clear() {
            objects.clear();
        }

        /*
            add object to the hittable ist
        */
        void add(shared_ptr<hittable> object) {
            objects.push_back(object);
        }
        
        /*
            determine if a ray hit any of the objects in the list

            @param r the ray
            @param ray_tmin the minimum parameter t in P = A + Bt for the ray
            @param ray_tmax the maximum parameter t in P = A + Bt for the ray
            @param rec the hit record that determines the details of the hit

            @return true if the ray hit any object, false otherwise
        */
        bool hit(const ray&r, interval ray_t, hit_record &rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            // we only "see" the closest object since it "blocks" the others behind it
            double closest_so_far = ray_t.max;

            // loop through the list and see if the ray hits anything
            for (const shared_ptr<hittable> &object : objects) {
                if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};

#endif