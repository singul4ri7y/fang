#include <env/cpu/asm/x86.h>
#include <env/cpu/gemm.h>

/* ================ KERNEL ================ */

/* Single precision GEMM 6x16 micro-kernel written in x86_64 (Haswell uArch)
   assembly. */
FANG_HOT void _fang_sgemm_6x16_ukernel(int k, float beta, float *restrict dest,
    int ld_dest, float alpha, float *restrict x, float *restrict y)
{
    const float one = 1.0f;

    _fang_begin_asm()

    /* It is a good idea to calculate `alpha * xy` first. */
    _vzeroall()

    _movq(rax, _v(x))  // `x`
    _movq(rbx, _v(y))  // `y`
    _xor(rcx, rcx)     // Zero out
    _movl(ecx, _v(k))  // `k`

    // TODO: Unroll to a factor.
    _label(._fang_sgemm_6x16_ukernel_kiter)
        _vmovaps(ymm12, _mem(rbx))              // `y`
        _vmovaps(ymm13, _mem(rbx, 0x20))        // `y + 8` (0x20, for float32)

        _vbroadcastss(ymm14, _mem(rax))         // `x`
        _vfmadd231ps(ymm0, ymm12, ymm14)
        _vfmadd231ps(ymm1, ymm13, ymm14)

        _vbroadcastss(ymm14, _mem(rax, 0x04))   // `x + 1` (0x04, for float32)
        _vfmadd231ps(ymm2, ymm12, ymm14)
        _vfmadd231ps(ymm3, ymm13, ymm14)

        _vbroadcastss(ymm14, _mem(rax, 0x08))   // `x + 2`
        _vfmadd231ps(ymm4, ymm12, ymm14)
        _vfmadd231ps(ymm5, ymm13, ymm14)

        _vbroadcastss(ymm14, _mem(rax, 0x0C))   // `x + 3`
        _vfmadd231ps(ymm6, ymm12, ymm14)
        _vfmadd231ps(ymm7, ymm13, ymm14)

        _vbroadcastss(ymm14, _mem(rax, 0x10))   // `x + 4`
        _vfmadd231ps(ymm8, ymm12, ymm14)
        _vfmadd231ps(ymm9, ymm13, ymm14)

        _vbroadcastss(ymm14, _mem(rax, 0x14))   // `x + 5`
        _vfmadd231ps(ymm10, ymm12, ymm14)
        _vfmadd231ps(ymm11, ymm13, ymm14)

        _add(rax, _c(0x18))  // `x += 6`, MR = 6
        _add(rbx, _c(0x40))  // `y += 16`, NR = 16

        _dec(rcx)
        _jnz(._fang_sgemm_6x16_ukernel_kiter)

    /* Store `one` to xmm15. */
    _vmovss(xmm15, _v(one))

    /* Apply `alpha`. */
    _vmovss(xmm12, _v(alpha))
    _vucomiss(xmm12, xmm15)  // Compare `alpha` against `one`
    /* No need to apply `alpha` if equal. */
    _je(.fang_sgemm_6x16_ukernel_lea)

    _vbroadcastss(ymm12, xmm12)
    _vmulps(ymm0, ymm0, ymm12)
    _vmulps(ymm1, ymm1, ymm12)
    _vmulps(ymm2, ymm2, ymm12)
    _vmulps(ymm3, ymm3, ymm12)
    _vmulps(ymm4, ymm4, ymm12)
    _vmulps(ymm5, ymm5, ymm12)
    _vmulps(ymm6, ymm6, ymm12)
    _vmulps(ymm7, ymm7, ymm12)
    _vmulps(ymm8, ymm8, ymm12)
    _vmulps(ymm9, ymm9, ymm12)
    _vmulps(ymm10, ymm10, ymm12)
    _vmulps(ymm11, ymm11, ymm12)

    _label(.fang_sgemm_6x16_ukernel_lea)
    /* Calculate addresses of `dest` beforehand. */
    _xor(rcx, rcx)
    _movl(ecx, _v(ld_dest))         // `ld_dest`
    _shl(rcx, _c(2))                // `ld_dest * 4`, float32 is 4-bytes

    _movq(rax, _v(dest))            // `dest`
    _lea(rbx, _mem(rax, 0x20))      // `dest + 8`
    _lea(rdx, _mem(rax, rcx, 1))    // `dest + 1 * ld_dest`
    _lea(rdi, _mem(rdx, 0x20))      // `dest + 1 * ld_dest + 8`
    _lea(r8, _mem(rax, rcx, 2))     // `dest + 2 * ld_dest`
    _lea(r9, _mem(r8, 0x20))        // `dest + 2 * ld_dest + 8`
    _lea(r10, _mem(r8, rcx, 1))     // `dest + 3 * ld_dest`
    _lea(r11, _mem(r10, 0x20))      // `dest + 3 * ld_dest + 8`
    _lea(r12, _mem(rax, rcx, 4))    // `dest + 4 * ld_dest`
    _lea(r13, _mem(r12, 0x20))      // `dest + 4 * ld_dest + 8`
    _lea(r14, _mem(r12, rcx, 1))    // `dest + 5 * ld_dest`
    _lea(r15, _mem(r14, 0x20))      // `dest + 5 * ld_dest + 8`

    /* Apply `beta`. */
    _vmovss(xmm12, _v(beta))
    _vxorps(xmm13, xmm13, xmm13)

    /* Compare `beta` against 0. */
    _vucomiss(xmm12, xmm13)

    /* If `beta` is 0.0, no need to load `dest`. */
    _je(._fang_sgemm_6x16_ukernel_done)

    /* Check if `beta` is one. If so, load and add `dest`, no need to multiply
       `beta`. */
    _vucomiss(xmm12, xmm15)
    /* No need to multiply `beta`. */
    _je(._fang_sgemm_6x16_ukernel_accm)

    _vbroadcastss(ymm12, xmm12)

    _vmovups(ymm13, _mem(rax))          // `dest`
    _vfmadd231ps(ymm0, ymm12, ymm13)
    _vmovups(ymm13, _mem(rbx))          // `dest + 8`
    _vfmadd231ps(ymm1, ymm12, ymm13)

    _vmovups(ymm13, _mem(rdx))          // `dest + ld_dest`
    _vfmadd231ps(ymm2, ymm12, ymm13)
    _vmovups(ymm13, _mem(rdi))          // `dest + ld_dest + 8`
    _vfmadd231ps(ymm3, ymm12, ymm13)

    _vmovups(ymm13, _mem(r8))           // `dest + 2 * ld_dest`
    _vfmadd231ps(ymm4, ymm12, ymm13)
    _vmovups(ymm13, _mem(r9))           // `dest + 2 * ld_dest + 8`
    _vfmadd231ps(ymm5, ymm12, ymm13)

    _vmovups(ymm13, _mem(r10))          // `dest + 3 * ld_dest`
    _vfmadd231ps(ymm6, ymm12, ymm13)
    _vmovups(ymm13, _mem(r11))          // `dest + 3 * ld_dest + 8`
    _vfmadd231ps(ymm7, ymm12, ymm13)

    _vmovups(ymm13, _mem(r12))          // `dest + 4 * ld_dest`
    _vfmadd231ps(ymm8, ymm12, ymm13)
    _vmovups(ymm13, _mem(r13))          // `dest + 4 * ld_dest + 8`
    _vfmadd231ps(ymm9, ymm12, ymm13)

    _vmovups(ymm13, _mem(r14))          // `dest + 5 * ld_dest`
    _vfmadd231ps(ymm10, ymm12, ymm13)
    _vmovups(ymm13, _mem(r15))          // `dest + 5 * ld_dest + 8`
    _vfmadd231ps(ymm11, ymm12, ymm13)

    /* Done applying `beta`. */
    _jmp(._fang_sgemm_6x16_ukernel_done)


    _label(._fang_sgemm_6x16_ukernel_accm)
    /* Accumulate `dest` to vector registers. */
    _vmovups(ymm12, _mem(rax))      // `dest`
    _vaddps(ymm0, ymm0, ymm12)
    _vmovups(ymm12, _mem(rbx))      // `dest + 8`
    _vaddps(ymm1, ymm1, ymm12)

    _vmovups(ymm12, _mem(rdx))      // `dest + ld_dest`
    _vaddps(ymm2, ymm2, ymm12)
    _vmovups(ymm12, _mem(rdi))      // `dest + ld_dest + 8`
    _vaddps(ymm3, ymm3, ymm12)

    _vmovups(ymm12, _mem(r8))       // `dest + 2 * ld_dest`
    _vaddps(ymm4, ymm4, ymm12)
    _vmovups(ymm12, _mem(r9))       // `dest + 2 * ld_dest + 8`
    _vaddps(ymm5, ymm5, ymm12)

    _vmovups(ymm12, _mem(r10))      // `dest + 3 * ld_dest`
    _vaddps(ymm6, ymm6, ymm12)
    _vmovups(ymm12, _mem(r11))      // `dest + 3 * ld_dest + 8`
    _vaddps(ymm7, ymm7, ymm12)

    _vmovups(ymm12, _mem(r12))      // `dest + 4 * ld_dest`
    _vaddps(ymm8, ymm8, ymm12)
    _vmovups(ymm12, _mem(r13))      // `dest + 4 * ld_dest + 8`
    _vaddps(ymm9, ymm9, ymm12)

    _vmovups(ymm12, _mem(r14))      // `dest + 5 * ld_dest`
    _vaddps(ymm10, ymm10, ymm12)
    _vmovups(ymm12, _mem(r15))      // `dest + 5 * ld_dest + 8`
    _vaddps(ymm11, ymm11, ymm12)


    _label(._fang_sgemm_6x16_ukernel_done)
    /* Write values to `dest`. */
    _vmovups(_mem(rax), ymm0)     // `dest`
    _vmovups(_mem(rbx), ymm1)     // `dest + 8`
    _vmovups(_mem(rdx), ymm2)     // `dest + ld_dest`
    _vmovups(_mem(rdi), ymm3)     // `dest + ld_dest + 8`
    _vmovups(_mem(r8), ymm4)      // `dest + 2 * ld_dest`
    _vmovups(_mem(r9), ymm5)      // `dest + 2 * ld_dest + 8`
    _vmovups(_mem(r10), ymm6)     // `dest + 3 * ld_dest`
    _vmovups(_mem(r11), ymm7)     // `dest + 3 * ld_dest + 8`
    _vmovups(_mem(r12), ymm8)     // `dest + 4 * ld_dest`
    _vmovups(_mem(r13), ymm9)     // `dest + 4 * ld_dest + 8`
    _vmovups(_mem(r14), ymm10)    // `dest + 5 * ld_dest`
    _vmovups(_mem(r15), ymm11)    // `dest + 5 * ld_dest + 8`

    _fang_end_asm(
        :  // No output
        : _fang_inop(k, m),
          _fang_inop(beta, m),
          _fang_inop(dest, m),
          _fang_inop(ld_dest, m),
          _fang_inop(alpha, m),
          _fang_inop(x, m),
          _fang_inop(y, m),
          _fang_inop(one, m)
        : "rax", "rbx", "rcx", "rdx", "rdi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "xmm0", "xmm1", "memory"
    );
}

/* ================ KERNEL ================ */
