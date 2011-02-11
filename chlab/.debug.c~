
#include <stdio.h>
#include <stdarg.h>

#include "debug.h"

char * CEX_thread_name = "no-name";

void
CEX_xprintf(char *frmt, ...)
{
        /* write to buffer to minimize issues with parallel IO */
        va_list va;
        char buffer[1024];

        va_start(va, frmt);
        vsnprintf(buffer, sizeof(buffer), frmt, va);
        va_end(va);
        fprintf(stderr, buffer);
        fflush(stderr);
}

/* "../stuff/name.c" -> "name.c" */
static const char *
fixFileName(const char *p)
{
        int i,l;

        l = strlen(p);
        for (i=l; i>0 && p[i-1] != '/'; i--);
        return p+i;
}

void
CEX_Fatal(const char *file, const char *func, int line,
                 const char *frmt, ...)
{
        va_list va;

        fprintf(stderr," *** CEX Fatal ***\n");
        fflush(stderr);
        fprintf(stderr,"In %.200s:%.200s:%.200s:%i\n", 
                CEX_thread_name, fixFileName(file), func, line);
        fflush(stderr);
        va_start(va, frmt);
        vfprintf(stderr, frmt, va);
        va_end(va);
        fprintf(stderr,"\n");
        fflush(stderr);
        abort();
}

