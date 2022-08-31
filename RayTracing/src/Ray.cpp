#include "Ray.h"

#include "ColorUtils.h"
#include "Object/Sphere.h"
#include "Object/HittableObjectList.h"
#include "Random.h"
#include "Material/Lambertian.h"
#include "Profiler/Profiler.h"

#include <limits>

Ray::Ray(const Ray::point3& origin, const Ray::vec3& direction)
	: m_Origin(origin), m_Direction(direction)
{
    
}

Ray::color Ray::At(float t) const
{
	return m_Origin + t * m_Direction;
}

float infinity = std::numeric_limits<double>::infinity();

Ray::color Ray::RayColor(Ray& ray, HittableObjectList& list, int32_t depth)
{

    if (depth <= 0)
        return color(0.0f, 0.0f, 0.0f);


    HitRecord hitRecord;
    if (list.Hit(ray, 0.001f, infinity, hitRecord))
    {
        Ray scattered;
        color attenuation;
        if (hitRecord.material_ptr->Scatter(ray, hitRecord, attenuation, scattered))
            return attenuation * RayColor(scattered, list, depth - 1);
        return color(0.0f, 0.0f, 0.0f);
        //point3 target = hitRecord.point + Random::RandomInHemisphere(hitRecord.normal);
        //vec3 n = hitRecord.point - vec3(0.0f, 0.0f, -1.0f);
        //n = n / glm::length(n);
        //return 0.5f * RayColor(Ray(hitRecord.point, target - hitRecord.point), list, depth - 1);// +color(1.0f));
    }
    //co cot = hit_sphere(point3(0.0f, 0.0f, -1.0f), 0.5, ray);
    //if (cot.t1 > 0.0f || cot.t2 > 0.0f)
    //{
    //    vec3 n = ray.At(cot.t1) - vec3(0.0f, 0.0f, -1.0f);
    //    n = n / glm::length(n);
    //    color c1 = 0.5f * (n + color(1.0f));
    //
    //    n = ray.At(cot.t2) - vec3(0.0f, 0.0f, -1.0f);
    //    n = n / glm::length(n);
    //    color c2 = 0.5f * (n + color(1.0f));
    //
    //    return ColorUtils::Vec4ToRGBABlendColor(glm::vec4(c2, 1.0f), glm::vec4(c1, 1.0f));
    //}
	vec3 unit_direction = ray.m_Direction / glm::length(ray.m_Direction);
	float t = 0.5f * (unit_direction.y + 1.0f);
	return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}