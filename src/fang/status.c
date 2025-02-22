#include "fang/status.h"

const char *fang_statstring(int status) {                      
    if (FANG_UNLIKELY(status > 0)) {                                                                   
        status = -status;                                                               
    }

    const char *ret;
    
    switch(-status) {                                                                    
        case FANG_OK:                                                                   
            ret = "Everything is OK... Successful operation";
            break;
        case FANG_INVID:                                                                
            ret = "Invalid ID";
            break;
        case FANG_NOMEM:
            ret = "No memory";
            break;
        case FANG_NOINFO:
            ret = "Could not retrieve information";
            break;
        case FANG_NOENV:                            
            ret = "No such Environment. Environment does not exist";
            break;
        case FANG_NTENS:                            
            ret = "Environment still in use by Tensors";            
            break;
        case FANG_INVENVTYP:                        
            ret = "Invalid Environment type";      
            break;
        case FANG_INVPCPU:                          
            ret = "Invalid physical CPU id";
            break;
        case FANG_INVPCOUNT:                        
            ret = "Invalid processor count";
            break;
        case FANG_ENVNOMATCH:                       
            ret = "Environment mismatch. Tensors do not belong to same Environment";
            break;
        case FANG_INVDIM:                           
            ret = "Invalid dimension";
            break;
        case FANG_INVSCTEN:                         
            ret = "Invalid scalar tensor";
            break;
        case FANG_INVTENTYP:                        
            ret = "Invalid tensor type";
            break;
        case FANG_INVDTYP:                          
            ret = "Invalid tensor data type";
            break;
        case FANG_UNSUPDTYP:                        
            ret = "Unsupported tensor data type";  
            break;
        case FANG_RANDOF:                           
            ret = "Difference overflow in tensor randomizer. Try reducing the range between low and high values.";
            break;
        case FANG_DESTINVDIM:                       
            ret = "Destination tensor dimension mismatch";          
            break;
        case FANG_NOBROAD:                          
            ret = "Tensor not broadcastable";      
            break;
        case FANG_INCMATDIM:                        
            ret = "Incompatible matrix dimensions in fang_ten_gemm()";
            break;
        default:                                    
            ret = "Unknown status";
            break;
    }

    return ret;
}                                                                                       