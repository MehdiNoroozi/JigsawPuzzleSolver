#! /bin/sh
# Copyright (C) 1997-2014 Free Software Foundation, Inc.
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

# Test sources listed in conditional.
# Report from Rob Savoye <rob@cygnus.com>, and Lars J. Aas.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_CONDITIONAL([ONE], [true])
AM_CONDITIONAL([TWO], [false])
AM_CONDITIONAL([THREE], [maybe])
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = targ

if ONE
SONE = one.c
else
SONE =
endif

if TWO
STWO = two.c
else
STWO =
endif

if THREE
STHREE = three.c
else
STHREE =
endif

targ_SOURCES = $(SONE) $(STWO) $(STHREE)
END

$ACLOCAL
$AUTOMAKE

# 'b top' so that
sed -n '
/[oO][bB][jJ][eE][cC][tT].* =/ {
  : loop
  /\\$/ {
    p
    n
    b loop
  }
  p
}' Makefile.in >produced

cat >expected << 'EOF'
@ONE_TRUE@am__objects_1 = one.$(OBJEXT)
@TWO_TRUE@am__objects_2 = two.$(OBJEXT)
@THREE_TRUE@am__objects_3 = three.$(OBJEXT)
am_targ_OBJECTS = $(am__objects_1) $(am__objects_2) $(am__objects_3)
targ_OBJECTS = $(am_targ_OBJECTS)
EOF

diff expected produced

:
