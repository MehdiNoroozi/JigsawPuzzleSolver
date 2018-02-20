#!/bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
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

# Regression test for an internal error when @LIBOBJS@ is used in
# a variable that is not defined in the same conditions as the _LDADD
# that uses it.
# Report from Bill Davidson.

. test-init.sh

cat >>configure.ac <<'EOF'
AC_PROG_CC
AC_LIBSOURCE([bar.c])
AM_CONDITIONAL([CASE], [:])
AC_OUTPUT
EOF

: >bar.c

cat >>Makefile.am <<'EOF'
COMMON_LIBS = @LIBOBJS@
bin_PROGRAMS = foo
if ! CASE
foo_LDADD = $(COMMON_LIBS)
endif
EOF

$ACLOCAL
$AUTOMAKE

:
