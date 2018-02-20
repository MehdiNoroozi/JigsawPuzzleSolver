#! /bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
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

# Make sure m4_included files are also scanned for definitions.
# Report from Phil Edwards.

. test-init.sh

cat >> configure.ac << 'END'
AM_PROG_LIBTOOL
AC_OUTPUT
END

echo 'm4_include([a.m4])' > acinclude.m4
echo 'm4_include([b.m4])' > a.m4

cat >b.m4 <<EOF
m4_include([c.m4])
AC_DEFUN([AM_PROG_LIBTOOL],
[AC_REQUIRE([SOMETHING])dnl
AC_REQUIRE([SOMETHING_ELSE])dnl
])

AC_DEFUN([SOMETHING])
EOF

echo 'm4_include([d.m4])' > c.m4
echo 'AC_DEFUN([SOMETHING_ELSE])' >d.m4

mkdir defs
echo 'AC_DEFUN([SOMETHING_ELSE])' >defs/e.m4
echo 'AC_DEFUN([ANOTHER_MACRO])' >defs/f.m4

cat >>Makefile.am<<\EOF
ACLOCAL_AMFLAGS = -I defs
testdist1: distdir
	test -f $(distdir)/acinclude.m4
	test -f $(distdir)/a.m4
	test -f $(distdir)/b.m4
	test -f $(distdir)/c.m4
	test -f $(distdir)/d.m4
	test ! -d $(distdir)/defs
testdist2: distdir
	test -f $(distdir)/acinclude.m4
	test -f $(distdir)/a.m4
	test -f $(distdir)/b.m4
	test -f $(distdir)/c.m4
	test -f $(distdir)/d.m4
	test ! -f $(distdir)/defs/e.m4
	test -f $(distdir)/defs/f.m4
EOF

$ACLOCAL -I defs

$FGREP acinclude.m4 aclocal.m4
# None of the following macro should be included.  acinclude.m4
# includes the first four, and the last two are not needed at all.
$FGREP a.m4 aclocal.m4 && exit 1
$FGREP b.m4 aclocal.m4 && exit 1
$FGREP c.m4 aclocal.m4 && exit 1
$FGREP d.m4 aclocal.m4 && exit 1
$FGREP defs/e.m4 aclocal.m4 && exit 1
$FGREP defs/f.m4 aclocal.m4 && exit 1

$AUTOCONF
$AUTOMAKE

./configure
$MAKE testdist1

cp aclocal.m4 aclocal.old
$sleep
echo 'AC_DEFUN([FOO], [ANOTHER_MACRO])' >> c.m4
$MAKE
# Because c.m4 has changed, aclocal.m4 must have been rebuilt.
is_newest aclocal.m4 aclocal.old
# However, since FOO is not used, f.m4 should not be included
# and the contents of aclocal.m4 should remain the same
diff aclocal.m4 aclocal.old

# If FOO where to be used, that would be another story, of course:
# f.m4 should be included
$sleep
echo FOO >> configure.ac
$MAKE
$FGREP defs/f.m4 aclocal.m4
$MAKE testdist2

# Make sure aclocal diagnose missing included files with correct 'file:line:'.
rm -f b.m4
$ACLOCAL 2>stderr && { cat stderr >&2; exit 1; }
cat stderr >&2
grep 'a\.m4:1: .*b\.m4.*does not exist' stderr

:
