#include "fang/status.h"

const char *fang_statstring(int status) {                      
    if (FANG_UNLIKELY(status > 0)) {                                                                   
        status = -status;                                                               
    }

    char *ret;
    
    switch(-status) {                                                                    
        case FANG_OK:                                                                   
            ret = "0 :: Everything is OK... Successful operation";
            break;
        case FANG_INVID:                                                                
            ret = "1 :: Invalid ID";
            break;
        case FANG_NOMEM:
            ret = "2 :: No memory";
            break;
        case FANG_NOINFO:
            ret = "3 :: Could not retrieve information";
            break;
        case FANG_NOENV:                            
            ret = "100 :: No such Environment. Environment does not exist";
            break;
        case FANG_NTENS:                            
            ret = "101 :: Environment still in use by Tensors";            
            break;
        case FANG_INVENVTYP:                        
            ret = "102 :: Invalid Environment type";      
            break;
        case FANG_INVPCPU:                          
            ret = "103 :: Invalid physical CPU id";
            break;
        case FANG_INVPCOUNT:                        
            ret = "104 :: Invalid processor count";
            break;
        case FANG_ENVNOMATCH:                       
            ret = "105 :: Environment mismatch. Tensors do not belong to same Environment";
            break;
        case FANG_INVDIM:                           
            ret = "201 :: Invalid dimension";
            break;
        case FANG_INVSCTEN:                         
            ret = "202 :: Invalid scalar tensor";
            break;
        case FANG_INVTENTYP:                        
            ret = "203 :: Invalid tensor type";
            break;
        case FANG_INVDTYP:                          
            ret = "204 :: Invalid tensor data type";
            break;
        case FANG_UNSUPDTYP:                        
            ret = "205 :: Unsupported tensor data type";  
            break;
        case FANG_RANDOF:                           
            ret = "206 :: Difference overflow in tensor randomizer. Try reducing the range between low and high values.";
            break;
        case FANG_DESTINVDIM:                       
            ret = "207 :: Destination tensor dimension mismatch";          
            break;
        case FANG_NOBROAD:                          
            ret = "218 :: Tensor not broadcastable";      
            break;
        case FANG_INCMATDIM:                        
            ret = "209 :: Incompatible matrix dimensions in fang_ten_gemm()";
            break;
        default:                                    
            ret = "Unknown status";
            break;
    }
    return ret;
}                                                                                       