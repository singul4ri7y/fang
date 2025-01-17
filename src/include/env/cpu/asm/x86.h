#ifndef FANG_CPU_ASM_X86_H
#define FANG_CPU_ASM_X86_H

/* Detect compiler. Clang is mostly compatible with GCC, hence need not to be
   separately handled. */
#if defined(__GNUC__) || defined(__clang__)

/* NOTE: Even though GCC and Clang both use AT&T syntax by default, the macros
 *   try to follow the Intel Assembly flavor by default as it's more
 *   intuitive. Hence, all the `dest` operand will be to the left. */


/* ================ HELPER MACROS ================ */

/* Helps with generalized macros combining multiple macros. */
#define _get_macro(_1, _2, _3, _4, NAME, ...)    \
NAME

/* ================ HELPER MACROS END ================ */


/* ================ ASSEMBLY CONTROL ================ */

/* Begin the extended inline assembly. */
#define _fang_begin_asm()         __asm__ volatile(

/* End extended inline assembly. */
#define _fang_end_asm(...)        __VA_ARGS__)

/* Use this to pass an input operand. */
#define _fang_inop(var, cons)     [var] #cons (var)

/* Use this to pass an output operand. */
#define _fang_outop(var, cons)    _fang_inop(var, cons)

/* Constant. */
#define _c(x)                     $##x

/* Access a variable/parameter outside of inline assembly region. */
#define _v(var)                   %[var]

/* ================ ASSEMBLY CONTROL END ================ */


/* ================ REGISTERS ================ */

/* General purpose registers. */
#define rax      %%rax
#define rbx      %%rbx
#define rcx      %%rcx
#define rdx      %%rdx
#define rsi      %%rsi
#define rdi      %%rdi
#define r8       %%r8
#define r9       %%r9
#define r10      %%r10
#define r11      %%r11
#define r12      %%r12
#define r13      %%r13
#define r14      %%r14
#define r15      %%r15

/* Lower 32-bit of general purpose registers. */
#define eax       %%eax
#define ebx       %%ebx
#define ecx       %%ecx
#define edx       %%edx

/* SSE registers. */
#define xmm0      %%xmm0
#define xmm1      %%xmm1
#define xmm2      %%xmm2
#define xmm3      %%xmm3
#define xmm4      %%xmm4
#define xmm5      %%xmm5
#define xmm6      %%xmm6
#define xmm7      %%xmm7
#define xmm8      %%xmm8
#define xmm9      %%xmm9
#define xmm10     %%xmm10
#define xmm11     %%xmm11
#define xmm12     %%xmm12
#define xmm13     %%xmm13
#define xmm14     %%xmm14
#define xmm15     %%xmm15

/* AVX2 Vector registers. */
#define ymm0     %%ymm0
#define ymm1     %%ymm1
#define ymm2     %%ymm2
#define ymm3     %%ymm3
#define ymm4     %%ymm4
#define ymm5     %%ymm5
#define ymm6     %%ymm6
#define ymm7     %%ymm7
#define ymm8     %%ymm8
#define ymm9     %%ymm9
#define ymm10    %%ymm10
#define ymm11    %%ymm11
#define ymm12    %%ymm12
#define ymm13    %%ymm13
#define ymm14    %%ymm14
#define ymm15    %%ymm15

/* ================ REGISTERS END ================ */


/* ================ INSTRUCTIONS ================ */

/* Convert instructions to string, makes writing assembly much more natural. */
#define _istring(...)                        #__VA_ARGS__ ";\n\t"

/* Zero out all the vector registers. */
#define _vzeroall()                          _istring(vzeroall)

/* Bitwise operations. */
#define _xor(dest, src)                      _istring(xor src, dest)
#define _shl(dest, n)                        _istring(shl n, dest)

/* Move operation. */
#define _mov(dest, src)                      _istring(mov src, dest)
/* Move a quad-word (64-bit). */
#define _movq(dest, src)                     _istring(movq src, dest)
/* Move a long-word (32-bit). */
#define _movl(dest, src)                     _istring(movl src, dest)

/* Effective memory manipulation. */
#define _mem1(base)                          (base)
#define _mem2(base, offset)                  offset (base)
#define _mem3(base, index, scale)            (base, index, scale)
#define _mem4(base, index, scale, offset)    offset (base, index, scale)

/* More general purpose memory access macro. */
#define _mem(...)    \
_get_macro(__VA_ARGS__, _mem4, _mem3, _mem2, _mem1)(__VA_ARGS__)

/* Load effective address. */
#define _lea(dest, mem)                      _istring(lea mem, dest)

/* Add instruction. */
#define _add(dest, src)                      _istring(add src, dest)

/* Increment and decrement instructions. */
#define _inc(src)                            _istring(inc src)
#define _dec(src)                            _istring(dec src)

/* Jump instructions. */
#define _jmp(label)                          _istring(jmp label)
#define _je(label)                           _istring(je label)
#define _jne(label)                          _istring(jne label)
#define _jl(label)                           _istring(jl label)
#define _jle(label)                          _istring(jle label)
#define _jz(label)                           _istring(jz label)
#define _jnz(label)                          _istring(jnz label)

/* Define a label. */
#define _label(label)                        _istring(label:)

/* SSE instructions. */
#define _xorps(dest, src)                    _istring(xorps src, dest)

/* AVX and AVX2 instructions. */
#define _vmovaps(dest, src)                  _istring(vmovaps src, dest)
#define _vmovups(dest, src)                  _istring(vmovups src, dest)
#define _vbroadcastss(dest, src)             _istring(vbroadcastss src, dest)
#define _vaddps(dest, src1, src2)            _istring(vaddps src1, src2, dest)
#define _vmulps(dest, src1, src2)            _istring(vmulps src1, src2, dest)
#define _vfmadd231ps(dest, src1, src2)       _istring(vfmadd231ps src1, src2, dest)
#define _vmovss(dest, src)                   _istring(vmovss src, dest)
#define _vucomiss(dest, src)                 _istring(vucomiss src, dest)
#define _vxorps(dest, src1, src2)            _istring(vxorps src1, src2, dest)

/* ================ INSTRUCTIONS ================ */

#elif defined(_MSC_VER)  // MSVC

// TODO: Enable MSVC x86 inline assembly support

#else
#error Compiler inline assembly not supported by Fang.
#endif  // Compiler detection

#endif  // FANG_CPU_ASM_X86_H
