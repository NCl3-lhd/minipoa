#ifndef MMPRIV_H
#define MMPRIV_H

#include <cstdint>
#include <cstddef>

typedef struct { uint64_t x, y; } mm128_t;
typedef struct { size_t n, m; mm128_t* a; } mm128_v;
void radix_sort_mm128x(mm128_t* beg, mm128_t* end);
static inline float mg_log2(float x) // NB: this doesn't work when x<2
{
	union { float f; uint32_t i; } z = { x };
	float log_2 = ((z.i >> 23) & 255) - 128;
	z.i &= ~(255 << 23);
	z.i += 127 << 23;
	log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
	return log_2;
}
#endif
