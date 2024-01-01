#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>

#include "hittable.h"

using std::shared_ptr;
using std::make_shared;

struct hittable_obj {
    hittable * object = nullptr;
    hittable_obj * next = nullptr;
};

/*
    class representing list of hittable objects
*/
class hittable_list : public hittable {
    public:
        // vector of hittable objects
        hittable_obj * first_obj = nullptr;
        hittable_obj * last_obj = nullptr;
        int size = 0;
        
        // constructors
        __host__ __device__ hittable_list() {}

        /*
            clear the list of hittable objects
        */
        __device__ void clear() {
            hittable_obj * curr = first_obj;

            while (curr != nullptr) {
                delete curr->object;
                hittable_obj * temp = curr->next;
                delete curr;
                curr = temp;
            }
        }

        /*
            add object to the hittable ist
        */
        __device__ void add(hittable * object) {
            if (object == nullptr) return;

            if (size == 0) {
                first_obj = new hittable_obj();
                first_obj->object = object;
                last_obj = first_obj;
                size++;
            } else {
                hittable_obj * temp = new hittable_obj();
                temp->object = object;
                last_obj->next = temp;
                last_obj = temp;
                size++;
            }
        }
        
        /*
            determine if a ray hit any of the objects in the list

            @param r the ray
            @param ray_tmin the minimum parameter t in P = A + Bt for the ray
            @param ray_tmax the maximum parameter t in P = A + Bt for the ray
            @param rec the hit record that determines the details of the hit

            @return true if the ray hit any object, false otherwise
        */
        __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            // we only "see" the closest object since it "blocks" the others behind it
            float closest_so_far = ray_t.max;

            // loop through the list and see if the ray hits anything
            hittable_obj * curr = first_obj;
            while (curr != nullptr) {
                hittable * object = curr->object;
                if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
                curr = curr->next;
            }

            return hit_anything;
        }
};

#endif