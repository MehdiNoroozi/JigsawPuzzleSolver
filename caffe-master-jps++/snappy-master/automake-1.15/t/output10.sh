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
# This is comparable to 'output9.sh', but testing Makefile rules.
# PR/411

. test-init.sh

cat >> configure.ac << END
AC_SUBST([FOO], [top])
AC_SUBST([BAR], [bot])
AC_CONFIG_FILES([a/top])
AC_CONFIG_FILES([a/bot])
AC_CONFIG_FILES([b/Makefile:a/top:b/Makefile.in:a/bot])
AC_OUTPUT
END

mkdir a
mkdir b

cat >Makefile.am <<\EOF
SUBDIRS = b
dist-hook:
	test ! -f $(distdir)/a/top
	test ! -f $(distdir)/a/bot
EOF

cat >b/Makefile.am <<\EOF
output:
	echo $(TOP)$(BOT) > ok
EOF

echo TOP=@FOO@ >a/top.in
echo BOT=@BAR@ >a/bot.in

$ACLOCAL
$AUTOCONF
$AUTOMAKE

mkdir build
cd build
../configure
cd b
$MAKE output
grep topbot ok
cd ..
$MAKE distcheck
