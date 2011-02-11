/* -*- Mode: c -*-
 * mem.c - Memory management utitlies
 *--------------------------------------------------------------------------
 * Copyright (C) 2009, Matthew Hagy (hagy@gatech.edu)
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "mem.h"
#include "debug.h"

#ifndef MACOSX
#  include <malloc.h>
#endif

void *
CEX_malloc(size_t bytes)
{
        void *p;
        p = malloc(bytes);
        if (p==NULL)
                Fatal("malloc failed of length %lu bytes", bytes);
        return p;
}

void *
CEX_realloc(void *ptr, size_t bytes)
{
        void *n;
        n = realloc(ptr, bytes);
        if (n==NULL)
                Fatal("realloc failed of length %lu bytes", bytes);
        return n;
}

void
CEX_free(void *p)
{
        free(p);
}

static inline size_t
align_min_malloc_bytes(size_t bytes, size_t alignment)
{
        return bytes + alignment - 1 + sizeof(void *);
}

/* given a pointer to a malloced region, we find the first block alignment
 * on given alignment, and then store the address of the malloced region 
 * in the word before the aligned region we return (standard aligned malloc) 
 */
static inline void *
align_malloced_bytes(void *base, size_t alignment)
{
        void *alignptr;
        alignptr = (void*)(((unsigned long)base + sizeof(void *) + alignment - 1) &
                           ~(alignment-1));
        *((void **)alignptr - 1) = base;
        return alignptr;
}

void *
CEX_aligned_malloc(size_t bytes, size_t alignment) 
{
        return align_malloced_bytes(CEX_malloc(align_min_malloc_bytes(bytes, alignment)),
                                    alignment);
}

void 
CEX_aligned_free(void *p)
{
        if (p) {
                CEX_free(*((void **)p - 1));
        }
}

void *
CEX_aligned_realloc(void *p, size_t bytes, size_t alignment) 
{
        //xprintf("aligned_realloc %p %lu %lu", p, bytes, alignment);
        return align_malloced_bytes(CEX_realloc(*((void **)p - 1),
                                                align_min_malloc_bytes(bytes, alignment)),
                                    alignment);
}

