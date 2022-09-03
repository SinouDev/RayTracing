#pragma once

#include "glm/glm.hpp"

#include <cstdlib>

#include "Walnut/Random.h"

class Random
{
public:

    static inline void Init()
    {
        s_Random.Init();
    }

    static inline double RandomDouble() 
    {
        // Returns a random real in [0,1).
        //float a = s_Random.Float();
        return s_Random.Float();// rand() / (RAND_MAX + 1.0f);
    }

    static inline double RandomDouble(double min, double max) 
    {
        // Returns a random real in [min,max).
        return min + (max - min) * RandomDouble();
    }

    static inline glm::vec3 RandomVec3() 
    {
        return { RandomDouble(), RandomDouble(), RandomDouble() };
    }

    static inline glm::vec3 RandomVec3(double min, double max)
    {
        return { RandomDouble(min, max), RandomDouble(min,  max), RandomDouble(min, max) };
    }

    static inline glm::vec3 RandomInUnitSphere()
    {
        //return Random::RandomVec3(-1.0, 1.0);
        while (true)
        {
            glm::vec3 p = Random::RandomVec3(-1.0, 1.0);
            if (glm::dot(p, p) >= 1.0f)
                continue;
            return p;
        }
    }

    static inline glm::vec3 RandomInUnitDisk()
    {
        //return Random::RandomVec3(-1.0, 1.0);
        while (true)
        {
            glm::vec3 p = glm::vec3(Random::RandomDouble(-1.0, 1.0), Random::RandomDouble(-1.0, 1.0), 0.0);
            if (glm::dot(p, p) >= 1.0f)
                continue;
            return p;
        }
    }

    static inline glm::vec3 RandomUnitVector()
    {
        glm::vec3 tmp = RandomInUnitSphere();
        return tmp / glm::length(tmp);
    }

    static inline glm::vec3 RandomInHemisphere(Ray::vec3& normal)
    {
        glm::vec3 in_unit_sphere = RandomInUnitSphere();
        if (glm::dot(in_unit_sphere, normal) > 0.0f)
        {
            return in_unit_sphere;
        }
        else
        {
            return -in_unit_sphere;
        }
    }

private:

    static Walnut::Random s_Random;

};

