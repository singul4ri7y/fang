#ifndef FANG_STATUS_H
#define FANG_STATUS_H

/* Only status of 0 refers OK. Generally, negative integer is returned in case
   of errors. */
#define FANG_ISOK(expr)    ((expr) >= 0)

/* Positive status. */
#define FANG_OK             0

/* Invalid ID. */
#define FANG_INVID          1

/* No memory. */
#define FANG_NOMEM          2

/* Could not retrieve information. */
#define FANG_NOINFO         3


/* ================ ENVIRONMENT ================ */

/* No such Environment. */
#define FANG_NOENV          100

/* Environment still in use by Tensors. */
#define FANG_NTENS          101

/* Invalid Environment type. */
#define FANG_INVENVTYP      102

/* Invalid physical CPU id. */
#define FANG_INVPCPU        103

/* Invalid processor count. */
#define FANG_INVPCOUNT      104

/* Environment mismatch. Tensors do not belong to same Environment. */
#define FANG_ENVNOMATCH     105

/* ================ ENVIRONMENT END ================ */


/* ================ TENSOR ================ */

/* Invalid dimension. */
#define FANG_INVDIM         201

/* Invalid scalar tensor. */
#define FANG_INVSCTEN       202

/* Invalid tensor type. */
#define FANG_INVTENTYP      203

/* Invalid tensor data type. */
#define FANG_INVDTYP        204

/* Unsupported tensor data type. */
#define FANG_UNSUPDTYP      205

/* Difference overflow in tensor randomizer. Try reducing the range between low
   and high values. */
#define FANG_RANDOF         206

/* Destination tensor dimension mismatch. */
#define FANG_DESTINVDIM     207

/* Tensors not broadcastable. */
#define FANG_NOBROAD        218

/* Incompatible matrix dimension in `fang_ten_gemm()`. */
#define FANG_INCMATDIM      209

/* ================ TENSOR END ================ */

#endif  // FANG_STATUS_H
