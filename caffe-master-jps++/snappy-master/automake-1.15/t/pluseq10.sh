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

# Test for += and backslashes.
# Reported by Ralf Corsepius.

. test-init.sh

cat >>configure.ac << 'END'
AM_CONDITIONAL([A], [true])
AM_CONDITIONAL([B], [false])
AC_OUTPUT
END

cat > Makefile.am << 'END'
foo =  0.h
if A
foo += a0.h \
  a1.h
foo += a2.h \
  a3.h
endif
if B
foo += b0.h \
  b1.h
endif

test:
	is $(foo) == 0.h a0.h a1.h a2.h a3.h
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE

./configure
$MAKE test

:
