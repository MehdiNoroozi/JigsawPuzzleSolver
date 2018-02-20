#! /bin/sh
# Copyright (C) 2003-2014 Free Software Foundation, Inc.
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

# Make sure aclocal does not require unused macros.

am_create_testdir=empty
. test-init.sh

cat > configure.ac << 'END'
AC_INIT
SOME_DEFS
END

mkdir m4
cat >m4/somedefs.m4 <<EOF
AC_DEFUN([SOME_DEFS], [
  m4_if([a], [a], [MACRO1], [MACRO2])
])
EOF

echo 'AC_DEFUN([MACRO1],)' >m4/macro1.m4
echo 'AC_DEFUN([MACRO2], [AC_REQUIRE([AM_UNUSED_MACRO])])' >m4/macro2.m4

$ACLOCAL -I m4 >output 2>&1 || { cat output; exit 1; }
test 0 -eq $(wc -l <output)
grep macro1.m4 aclocal.m4
grep macro2.m4 aclocal.m4 && exit 1

:
