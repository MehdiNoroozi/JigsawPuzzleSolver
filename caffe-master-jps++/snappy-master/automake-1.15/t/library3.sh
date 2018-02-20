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

# Make sure Automake simplify conditions in diagnostics.

. test-init.sh

cat >>configure.ac <<EOF
AC_PROG_CC
AM_CONDITIONAL([A], [:])
AM_CONDITIONAL([B], [:])
AM_CONDITIONAL([C], [:])
AM_CONDITIONAL([D], [:])
EOF

cat > Makefile.am << 'END'
if A
if !B
  RANLIB = anb
else
  RANLIB = ab
endif
endif
if C
  RANLIB = c
endif
if !C
if D
  RANLIB = ncd
endif
endif
EXTRA_LIBRARIES = libfoo.a
END

$ACLOCAL
AUTOMAKE_fails
grep '^Makefile.am:.*:   !A and !C and !D$' stderr
# Is there only one missing condition?
test $(grep -c ':   !' stderr) -eq 1

:
