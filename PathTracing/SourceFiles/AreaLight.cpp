#include "AreaLight.h"

AreaLight::AreaLight(const float3& p, const float3& i)
{
	Light(p, i);
	normal = make_float3(0, -1, 0);
	u = make_float3(1, 0, 0);
	v = make_float3(0, 0, 1);
	length = 100;
}
