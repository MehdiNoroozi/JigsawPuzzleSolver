#! /bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Make sure we reuse variables whenever possible, to limit
# combinational explosion.  (This test is named after the &subobjname
# sub in Automake).

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_CONDITIONAL([FOO1], [some test])
AM_CONDITIONAL([FOO2], [some test])
AM_CONDITIONAL([FOO3], [some test])
AC_OUTPUT
END

cat > Makefile.am << 'END'
noinst_PROGRAMS = c d

if FOO1
A1=a1.c
endif

if FOO2
A2=a2.c
endif

if FOO3
A3=a3.c
endif

B=$(A1) $(A2) $(A3)

c_SOURCES=$(B)
d_SOURCES=$(B)
END

$ACLOCAL
$AUTOMAKE -a

# Sanity check: make sure am_c_OBJECTS and am_d_OBJECTS are used
# in the Makefile.  (This is an internal detail, so better make
# sure we update this test if the naming changes in the future.)
grep '^am_c_OBJECTS = ' Makefile.in
grep '^am_d_OBJECTS = ' Makefile.in

# Now the actual test.  Are both values equal?
cobj=$(sed -n '/^am_c_OBJECTS = / {
                 s/.* = \(.*\)$/\1/
                 p
               }' Makefile.in)
dobj=$(sed -n '/^am_d_OBJECTS = / {
                 s/^.* = \(.*\)$/\1/
                 p
              }' Makefile.in)
test "$cobj" = "$dobj"

:
