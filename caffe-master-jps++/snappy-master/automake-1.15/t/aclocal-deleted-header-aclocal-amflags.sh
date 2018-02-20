#! /bin/sh
# Copyright (C) 2007-2014 Free Software Foundation, Inc.
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

# Make sure the "deleted header file" issue is fixed wrt. aclocal.m4
# dependencies.
# NOTE: this test works by using the obsolete 'ACLOCAL_AMFLAGS' make
# variable; see sister test 'aclocal-deleted-header.sh' for a modern
# equivalent.

. test-init.sh

cat >>configure.ac <<EOF
FOO
AC_OUTPUT
EOF

cat >foo.m4 <<EOF
AC_DEFUN([FOO], [AC_SUBST([GREPFOO])])
EOF

cat >bar.m4 <<EOF
AC_DEFUN([BAR], [AC_SUBST([GREPBAR])])
EOF

cat >Makefile.am <<EOF
ACLOCAL_AMFLAGS = -I .
EOF

$ACLOCAL -I .
$AUTOMAKE
$AUTOCONF

./configure

$MAKE
grep GREPFOO Makefile
grep GREPBAR Makefile && exit 1

sed 's/FOO/BAR/' < configure.ac > t
mv -f t configure.ac
rm -f foo.m4

$MAKE
grep GREPFOO Makefile && exit 1
grep GREPBAR Makefile

:
