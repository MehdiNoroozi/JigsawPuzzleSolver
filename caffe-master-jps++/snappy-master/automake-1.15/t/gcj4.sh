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

# Make sure dependency tracking works for Java.

required=gcj
. test-init.sh

cat >> configure.ac << 'END'
AM_PROG_GCJ
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = convert
convert_SOURCES = convert.java
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure >stdout || { cat stdout; exit 1; }
cat stdout

# Configure must be checking the dependency style of gcj ...
grep 'dependency style of gcj' stdout >filt
cat filt

# ... only once.
test $(wc -l < filt) = 1

# Accept any outcome but 'none'
# (at the time of writing it should be gcc or gcc3).
grep -v none filt

:
