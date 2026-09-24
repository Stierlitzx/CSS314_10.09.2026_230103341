#include <cpuid.h>
#include <stdio.h>

int main(void)
{
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(0x80000005U, &eax, &ebx, &ecx, &edx) == 0)
        return 1;

    printf("L1 data cache line size: %u bytes\n", ecx & 0xffU);
    return 0;
}
