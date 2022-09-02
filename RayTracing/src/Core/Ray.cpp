#include "Ray.h"

#include "../Core/Utils.h"
#include "../Object/Sphere.h"
#include "../Object/HittableObjectList.h"
#include "../Random.h"
#include "../Material/Lambertian.h"
#include "../Profiler/Profiler.h"

#include "Renderer.h"

#include <limits>

Ray::Ray(const point3& origin, const vec3& direction, const color& backgroundColor, const color& backgroundColor1)
	: m_Origin(origin), m_Direction(direction), m_RayBackgroundColor(backgroundColor), m_RayBackgroundColor1(backgroundColor1)
{
    m_RayBackgroundColor = Renderer::GetRayBackgroundColor();
    m_RayBackgroundColor1 = Renderer::GetRayBackgroundColor1();
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

    //vec3 lightDir = glm::normalize(ray.m_LightDir);

    //float d = 1.0f;

    HitRecord hitRecord;
    if (list.Hit(ray, 0.001f, infinity, hitRecord))
    {
        
        Ray scattered(ray.GetOrigin());
        color attenuation;
        if (hitRecord.material_ptr->Scatter(ray, hitRecord, attenuation, scattered))
        {
            color c = attenuation * RayColor(scattered, list, depth - 1);
            //d = glm::max(glm::dot(glm::normalize(hitRecord.point), -lightDir), 0.0f);
            


            return c;// *d;
        }
            
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
    vec3 unit_direction = Utils::UnitVec(ray.m_Direction);// / glm::length(ray.m_Direction);
	float t = 0.5f * (unit_direction.y + 1.0f);
	return (1.0f - t) * ray.m_RayBackgroundColor1 + t * ray.m_RayBackgroundColor;
}