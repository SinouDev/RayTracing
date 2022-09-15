#include "Ray.h"

#include "Object/Sphere.h"
#include "Object/HittableObject.h"
#include "Material/Lambertian.h"

#include "Renderer.h"

using Utils::Math::Point3;
using Utils::Math::Vec3;
using Utils::Math::Color3;
using Utils::Math::Color4;

Ray::Ray(const Point3& origin, const Vec3& direction, float time, const Color3& backgroundColor, const Color3& backgroundColor1)
	: m_Origin(origin), m_Direction(direction), m_Time(time), m_RayBackgroundColor(backgroundColor), m_RayBackgroundColor1(backgroundColor1)
{
    m_RayBackgroundColor = Renderer::GetRayBackgroundColor();
    m_RayBackgroundColor1 = Renderer::GetRayBackgroundColor1();
}

Point3 Ray::At(float t) const
{
	return m_Origin + t * m_Direction;
}

Color3 get_background(const Ray& ray)
{
    Vec3 unit_direction = Utils::Math::UnitVec(ray.m_Direction);// / Utils::Math::Q_Length(ray.m_Direction);
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * ray.m_RayBackgroundColor1 + t * ray.m_RayBackgroundColor;
}

Color4 Ray::RayColor(const Ray& ray, const Color3& backgroundColor, const HittableObject& list, int32_t depth)
{

    if (depth <= 0)
        return Color4(0.0f, 0.0f, 0.0f, 1.0f);

    //Vec3 lightDir = Utils::Math::UnitVec(ray.m_LightDir);

    //float d = 1.0f;

    HitRecord hitRecord;
    if (!list.Hit(ray, 0.001f, Utils::Math::infinity, hitRecord))
        return Color4(get_background(ray), 1.0f);
    //{
        
    Ray scattered;
    Color4 attenuation;
    Color3 emitted = hitRecord.material_ptr->Emitted(hitRecord.coord, hitRecord.point);
    if (!hitRecord.material_ptr->Scatter(ray, hitRecord, attenuation, scattered))
    {
        return Color4(emitted, 1.0f);
        //color c = ;
        //d = Utils::Math::Max(Utils::Math::Dot(Utils::Math::UnitVec(hitRecord.point), -lightDir), 0.0f);

    }

    if(s_SimpleRay)
        return attenuation;

    return Color4(emitted, 0.0f) + attenuation * RayColor(scattered, backgroundColor, list, depth - 1);// *d;
    //}
    //return Color3(0.0f, 0.0f, 0.0f);
    //Point3 target = hitRecord.point + Random::RandomInHemisphere(hitRecord.normal);
    //Vec3 n = hitRecord.point - Vec3(0.0f, 0.0f, -1.0f);
    //n = n / Utils::Math::Q_Length(n);
    //return 0.5f * RayColor(Ray(hitRecord.point, target - hitRecord.point), list, depth - 1);// +color(1.0f));
    //}
    //co cot = hit_sphere(Point3(0.0f, 0.0f, -1.0f), 0.5, ray);
    //if (cot.t1 > 0.0f || cot.t2 > 0.0f)
    //{
    //    Vec3 n = ray.At(cot.t1) - Vec3(0.0f, 0.0f, -1.0f);
    //    n = n / Utils::Math::Q_Length(n);
    //    Color3 c1 = 0.5f * (n + Color3(1.0f));
    //
    //    n = ray.At(cot.t2) - Vec3(0.0f, 0.0f, -1.0f);
    //    n = n / Utils::Math::Q_Length(n);
    //    Color3 c2 = 0.5f * (n + Color3(1.0f));
    //
    //    return Utils::Color::Vec4ToRGBABlendColor(Color4(c2, 1.0f), Color4(c1, 1.0f));
    //}
    //Vec3 unit_direction = Utils::Math::UnitVec(ray.m_Direction);// / Utils::Math::Q_Length(ray.m_Direction);
	//float t = 0.5f * (unit_direction.y + 1.0f);
	//return (1.0f - t) * ray.m_RayBackgroundColor1 + t * ray.m_RayBackgroundColor;
}