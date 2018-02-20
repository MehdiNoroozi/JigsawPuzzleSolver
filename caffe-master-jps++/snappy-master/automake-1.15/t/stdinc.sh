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

# Test to make sure the standard include order is stable.
# Report by Kent Boortz.

required=cc
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_CONFIG_HEADERS([sub/config.h])
AC_CONFIG_FILES([sub/bar.h])
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = foo
foo_SOURCES = foo.c
BUILT_SOURCES = bar.h
END

mkdir sub

cat >foo.c <<'END'
#include <config.h>
#include <bar.h>
int main() { return bar (); }
END
cat >bar.h <<'END'
int bar () { return 0; }
END
cat >sub/bar.h.in <<'END'
choke me
END

$ACLOCAL
$AUTOCONF
$AUTOHEADER
$AUTOMAKE

mkdir build
cd build
../configure -C
$MAKE

cd ..
./configure -C
$MAKE
