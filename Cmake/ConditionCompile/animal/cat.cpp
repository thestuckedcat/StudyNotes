# include "cat.h"

#ifdef USE_CATTWO
  # include "cattwo.h"
#endif



std::string Cat::barking(){
    #ifdef USE_CATTWO
        return cattwo::two();
    #else
        return "mewo mewo";
    #endif
  }