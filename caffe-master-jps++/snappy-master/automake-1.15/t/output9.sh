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

# Make sure an AC_CONFIG_FILES can have an AC_CONFIG_FILES output as input.

. test-init.sh

cat >> configure.ac << END
AC_CONFIG_FILES([a/mid.in:a/input.in.in])
AC_CONFIG_FILES([b/out:a/mid.in])
AC_CONFIG_FILES([a/Makefile b/Makefile])
AC_OUTPUT
END

mkdir a
mkdir b


cat >Makefile.am <<\EOF
SUBDIRS = a b
dist-hook:
	test -f $(distdir)/a/input.in.in
	test ! -f $(distdir)/a/mid.in
	if test ! -f check; then :; else : > ok; fi
EOF

: >a/Makefile.am
: >b/Makefile.am

echo foo >a/input.in.in

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing

./configure
: > check
$MAKE distcheck
test -f ok
