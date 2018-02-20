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

# Another check for README-alpha support.  This time it is requested
# from configure.ac.  Report from Akim Demaille.

. test-init.sh

cat > configure.ac << 'END'
AC_INIT([alpha], [1.0b])
AM_INIT_AUTOMAKE([readme-alpha])
AC_CONFIG_FILES([Makefile sub/Makefile])
AC_OUTPUT
END

cat > Makefile.am << 'END'
SUBDIRS = sub
check-local: distdir
	test -f $(distdir)/README-alpha
	test -f $(distdir)/sub/README
	test ! -f $(distdir)/sub/README-alpha
	: > works
END

mkdir sub
: > sub/Makefile.am

: > README-alpha
: > sub/README-alpha
: > sub/README

# Gnits stuff.
: > INSTALL
: > NEWS
: > README
: > COPYING
: > AUTHORS
: > ChangeLog
: > THANKS

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE check
test -f works
