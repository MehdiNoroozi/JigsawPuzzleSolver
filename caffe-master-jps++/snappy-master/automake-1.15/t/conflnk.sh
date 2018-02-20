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

# Test to make sure links created by AC_CONFIG_LINKS get removed with
# 'make distclean'

. test-init.sh

echo 'SUBDIRS = sdir' > Makefile.am
: > src
mkdir sdir
: > sdir/Makefile.am
: > sdir/src2
mkdir sdir-no-make

cat >>configure.ac << 'EOF'
AC_CONFIG_FILES([sdir/Makefile])
AC_CONFIG_LINKS([dest:src])
AC_CONFIG_LINKS([dest2:src])
AC_CONFIG_LINKS([sdir/dest3:src])
AC_CONFIG_LINKS([dest4:sdir/src2])
AC_CONFIG_LINKS([sdir/dest5:sdir/src2 sdir-no-make/dest6:src])
AC_OUTPUT
EOF

$ACLOCAL
$AUTOMAKE
$AUTOCONF
./configure

# Make sure nothing is deleted by 'make clean'
$MAKE clean

test -r dest
test -r dest2
test -r sdir/dest3
test -r dest4
test -r sdir/dest5
test -r sdir-no-make/dest6
test -f src
test -f sdir/src2

# Make sure the links are deleted by 'make distclean' and the original files
# are not.
$MAKE distclean

test -f src
test -f sdir/src2

test -r dest && exit 1
test -r dest2 && exit 1
test -r sdir/dest3 && exit 1
test -r dest4 && exit 1
test -r sdir/dest5 && exit 1
test -r sdir-no-make/dest6 && exit 1

:
