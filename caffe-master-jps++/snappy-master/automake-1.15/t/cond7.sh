#! /bin/sh
# Copyright (C) 1998-2014 Free Software Foundation, Inc.
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

# Test for an odd conditional bug.  Report from Pavel Roskin.

. test-init.sh

cat >> configure.ac << 'END'
compat=yes
AM_CONDITIONAL([Compatible], [test x$compat = xyes])
AC_OUTPUT
END

cat > Makefile.am << 'END'
if Compatible
abdir = none
ab_HEADERS = \
        minus.h
endif
END

$ACLOCAL
$AUTOMAKE

grep '^[^#].*002' Makefile.in && exit 1
exit 0
