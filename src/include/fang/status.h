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

/* Cannot create thread/task. */
#define FANG_NOTASK         205

/* Cannot pin task kernel to a specific processor. */
#define FANG_NOPIN          206

/* Difference overflow in tensor randomizer. Try reducing the range between low
   and high values. */
#define FANG_RANDOF         207

/* ================ TENSOR END ================ */

#endif  // FANG_STATUS_H
