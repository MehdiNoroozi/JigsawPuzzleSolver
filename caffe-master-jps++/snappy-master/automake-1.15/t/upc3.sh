#! /bin/sh
# Copyright (C) 2006-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Test that C and Unified Parallel C link safely.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_PROG_UPC
AC_OUTPUT
END

cat > hello.upc << 'END'
#include <stdio.h>
#include <upc.h>
#include "message.h"
int
main (void)
{
  printf ("Thread %d says, '%s'\n", MYTHREAD, message());
  return 0;
}
END

cat > message.h << 'END'
const char *message (void);
END

cat > mix.c << 'END'
#include "message.h"
const char *
message (void)
{
  return "Hello.";
}
END

cat > Makefile.am << 'END'
bin_PROGRAMS = hello
hello_SOURCES = hello.upc mix.c message.h
hello_LDADD = -lm
END

$ACLOCAL
$AUTOMAKE
$AUTOCONF

./configure || exit $?
$MAKE distcheck

:
