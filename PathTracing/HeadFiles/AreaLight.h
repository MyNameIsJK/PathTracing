#pragma once
#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include "Light.h"

class AreaLight
{
public:
	float length;
	float3 normal;
	float3 u;
	float3 v;
public:
	AreaLight(const float3& p, const float3& i);
};

#endif