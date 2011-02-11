/* -*- Mode: c -*-
 * dbg.h - Minimal debuggging support
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

#ifndef DEBUG_H
#define DEBUG_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opt.h"

extern char * CEX_thread_name;

void CEX_xprintf(char *frmt, ...) GCC_ATTRIBUTE((format (printf, 1, 2)));

#ifndef NO_XPRINTF
# define xprintf(msg, ARGS...)                                          \
        CEX_xprintf("cex:%s: " msg "\n", CEX_thread_name, ## ARGS)
#else /* NO_XPRINTF */
# define xprintf(msg, ARGS...) do{}while(0)
#endif /* NO_XPRINTF */

void CEX_Fatal(const char *file, const char *func, int line,
                 const char *frmt, ...)
        GCC_ATTRIBUTE((format (printf, 4, 5)))
        GCC_ATTRIBUTE((noreturn));

#define Fatal(FRMT, ARGS...)                            \
        CEX_Fatal(__FILE__, __FUNCTION__, __LINE__, FRMT, ## ARGS)

#endif /* DEBUG_H */
