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

# Make sure included files in parent directory are distributed.

. test-init.sh

cat >> configure.ac << 'END'
AC_CONFIG_FILES([sub/Makefile])
AC_OUTPUT
END

cat > Makefile.am << 'END'
SUBDIRS = sub
test: distdir
	test -f $(distdir)/foo
	test -f $(distdir)/bar
	test 2 -gt `find $(distdir)/sub -type d | wc -l`
END

: > foo
: > bar

mkdir sub
cat > sub/Makefile.am << 'END'
include $(top_srcdir)/foo
include ../bar
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE
# Use --srcdir with an absolute path because it's harder
# to support in 'distdir'.
./configure --srcdir "$(pwd)"
$MAKE test

:
